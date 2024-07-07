import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tqdm
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
from torchvision.utils import save_image

from models.vqmodel import VQModel
from utils.logger import get_logger
from utils.misc import image_norm_to_float, instantiate_from_config, amortize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to inference configuration file')
    parser.add_argument('-vw', '--vqmodel_weights', type=str, required=True, help='Path to pretrained vqmodel weights')
    parser.add_argument('-pw', '--prior_weights', type=str, required=True, help='Path to pretrained prior weights')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory saving samples')
    parser.add_argument('--bspp', type=int, default=500, help='Batch size on each process')
    parser.add_argument('--topk', type=int, default=100, help='Top-k sampling in prior model')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(use_tqdm_handler=True, is_main_process=accelerator.is_main_process)

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD MODEL
    encoder = instantiate_from_config(conf.vqmodel.encoder)
    decoder = instantiate_from_config(conf.vqmodel.decoder)
    quantizer = instantiate_from_config(conf.vqmodel.quantizer)
    vqmodel = VQModel(
        encoder=encoder, decoder=decoder, quantizer=quantizer,
        z_channels=conf.vqmodel.z_channels, codebook_dim=conf.vqmodel.codebook_dim,
    ).eval().to(device)
    prior = instantiate_from_config(conf.prior).eval()

    # LOAD WEIGHTS
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')
    logger.info(f'Number of parameters of prior model: {sum(p.numel() for p in prior.parameters()):,}')
    ckpt = torch.load(args.vqmodel_weights, map_location='cpu')
    vqmodel.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load vqmodel from {args.vqmodel_weights}')
    ckpt = torch.load(args.prior_weights, map_location='cpu')
    prior.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load prior model from {args.prior_weights}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    vqmodel, prior = accelerator.prepare(vqmodel, prior)  # type: ignore
    unwrapped_vqmodel = accelerator.unwrap_model(vqmodel)
    unwrapped_prior = accelerator.unwrap_model(prior)

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    idx = 0
    fm_size = conf.data.img_size // conf.vqmodel.downsample_factor  # feature map size
    with torch.no_grad():
        bslist = amortize(args.n_samples, args.bspp * accelerator.num_processes)
        for bs in tqdm.tqdm(bslist, desc='Sampling', disable=not accelerator.is_main_process):
            bspp = min(args.bspp, math.ceil(bs / accelerator.num_processes))
            idxmap = unwrapped_prior.sample(B=bspp, L=fm_size ** 2, topk=args.topk)
            z = unwrapped_vqmodel.codebook(idxmap).reshape(bspp, fm_size, fm_size, -1).permute(0, 3, 1, 2)
            samples = unwrapped_vqmodel.decode(z)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                    idx += 1
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
