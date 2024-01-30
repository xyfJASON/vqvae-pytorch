import os
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from accelerate.utils import DistributedDataParallelKwargs

from utils.logger import StatusTracker, get_logger
from utils.misc import get_time_str, check_freq, get_data_generator
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
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
    # ARGS & CONF
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            conf_yaml=OmegaConf.to_yaml(conf),
            exist_ok=args.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=conf.train.print_freq,
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
    train_set = instantiate_from_config(conf.data)
    valid_set = instantiate_from_config(conf.data_valid)
    batch_size_per_process = conf.train.batch_size // accelerator.num_processes
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size_per_process,
        shuffle=True, drop_last=True, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {conf.train.batch_size}')
    logger.info('=' * 50)

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(conf.model)
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
        # load model
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {ckpt_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        # save model
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model.pt'))
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # type: ignore

    accelerator.wait_for_everyone()

    def _discard_labels(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def run_step(_batch):
        x = _discard_labels(_batch).float()
        out = model(x)

        loss_rec = F.mse_loss(out['decx'], x)
        loss_vq = F.mse_loss(out['z'].detach(), out['quantized_z'])
        loss_commit = F.mse_loss(out['z'], out['quantized_z'].detach())
        if conf.model.params.codebook_update == 'kmeans':
            loss_vq = 0.0

        loss = loss_rec + loss_vq + conf.train.coef_commit * loss_commit
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        if conf.model.params.codebook_update == 'kmeans':
            codebook_num = conf.model.params.codebook_num
            codebook_dim = conf.model.params.codebook_dim
            flat_z = out['z'].detach().permute(0, 2, 3, 1).reshape(-1, codebook_dim)
            indices_count = torch.bincount(out['indices'], minlength=codebook_num)
            new_sumz = torch.zeros((codebook_num, codebook_dim), device=device)
            new_sumz.scatter_add_(dim=0, index=out['indices'][:, None].repeat(1, codebook_dim), src=flat_z)

            new_sumz = accelerate.utils.reduce(new_sumz, reduction='sum')
            new_sumn = accelerate.utils.reduce(indices_count, reduction='sum')
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.update_codebook(new_sumz, new_sumn)

        return dict(
            loss_rec=loss_rec.item(),
            loss_vq=loss_vq.item(),
            loss_commit=loss_commit.item(),
            perplexity=out['perplexity'].item(),
            lr=optimizer.param_groups[0]['lr'],
        )

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath):
        unwrapped_model = accelerator.unwrap_model(model)
        X = torch.stack([_discard_labels(valid_set[i]) for i in range(32)], dim=0).float().to(device)
        recX = unwrapped_model.reconstruct(X)
        C, H, W = recX.shape[1:]
        show = torch.stack((X, recX), dim=1).reshape(-1, C, H, W)
        save_image(show, savepath, nrow=8, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


if __name__ == '__main__':
    main()
