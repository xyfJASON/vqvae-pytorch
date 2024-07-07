import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from omegaconf import OmegaConf

import torch
import accelerate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.vqmodel import VQModel

from utils.data import load_data
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config
from utils.misc import get_time_str, check_freq, get_data_generator, discard_label


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-r', '--resume', type=str,
        help='Resume from a checkpoint. Could be a path or `best` or `latest`',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), time_str=args.time_str,
            exist_ok=args.resume is not None, no_interaction=args.no_interaction,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, exp_dir=exp_dir, print_freq=conf.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    if conf.train.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f'Batch size should be divisible by number of processes, '
            f'get {conf.train.batch_size} % {accelerator.num_processes} != 0'
        )
    bspp = conf.train.batch_size // accelerator.num_processes
    train_set = load_data(conf.data, split='train')
    train_loader = DataLoader(
        dataset=train_set, batch_size=bspp,
        shuffle=True, drop_last=True, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # LOAD PRETRAINED VQMODEL
    encoder = instantiate_from_config(conf.vqmodel.encoder)
    decoder = instantiate_from_config(conf.vqmodel.decoder)
    quantizer = instantiate_from_config(conf.vqmodel.quantizer)
    vqmodel = VQModel(
        encoder=encoder, decoder=decoder, quantizer=quantizer,
        z_channels=conf.vqmodel.z_channels, codebook_dim=conf.vqmodel.codebook_dim,
    ).requires_grad_(False).eval().to(device)
    weights = torch.load(conf.vqmodel.pretrained, map_location='cpu')
    vqmodel.load_state_dict(weights['model'])
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load pretrained VQModel from {conf.vqmodel.pretrained}')

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(conf.model)
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    logger.info(f'Number of parameters of model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # RESUME TRAINING
    step = 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt_model = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(resume_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {resume_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(resume_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # type: ignore
    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone()

    # TRAINING FUNCTIONS
    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model.pt'))
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    def run_step(batch):
        x = discard_label(batch).float()
        idx = vqmodel.encode(x)['indices']  # (B * H * W)
        idx = idx.reshape(x.shape[0], -1)   # (B, H * W)

        preds = model(idx)                  # (B, H * W + 1, C)
        preds = preds[:, :-1, :]            # (B, H * W, C)
        loss = F.cross_entropy(preds.reshape(-1, preds.shape[-1]), idx.reshape(-1))

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        return dict(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath):
        fm_size = conf.data.img_size // conf.vqmodel.downsample_factor
        idx = unwrapped_model.sample(B=64, L=fm_size ** 2, topk=100)
        z = vqmodel.codebook(idx).reshape(64, fm_size, fm_size, -1).permute(0, 3, 1, 2)
        samples = vqmodel.decode(z).clamp(-1, 1)
        save_image(samples, savepath, nrow=8, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        _batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(_batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()
        # validate
        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


if __name__ == '__main__':
    main()
