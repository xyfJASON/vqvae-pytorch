import os
import math
import tqdm
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
from torchvision.utils import save_image

from utils.logger import get_logger
from utils.misc import image_norm_to_float, instantiate_from_config, amortize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=8888,
        help='Set random seed',
    )
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to inference configuration file',
    )
    parser.add_argument(
        '-vw', '--vqvae_weights', type=str, required=True,
        help='Path to pretrained vqvae weights',
    )
    parser.add_argument(
        '-tw', '--transformer_weights', type=str, required=True,
        help='Path to pretrained transformer prior weights',
    )
    parser.add_argument(
        '--n_samples', type=int, required=True,
        help='Number of samples',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--batch_size', type=int, default=500,
        help='Batch size on each process. Sample by batch is much faster',
    )
    parser.add_argument(
        '--topk', type=int, default=100,
        help='Top-k sampling in transformer',
    )
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
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD MODEL
    vqvae = instantiate_from_config(conf.vqvae)
    transformer = instantiate_from_config(conf.transformer)

    # LOAD WEIGHTS
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    ckpt = torch.load(args.vqvae_weights, map_location='cpu')
    vqvae.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load vqvae from {args.vqvae_weights}')
    ckpt = torch.load(args.transformer_weights, map_location='cpu')
    transformer.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load transformer from {args.transformer_weights}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    vqvae = accelerator.prepare(vqvae)
    transformer = accelerator.prepare(transformer)
    vqvae.eval()
    transformer.eval()

    accelerator.wait_for_everyone()

    @torch.no_grad()
    def sample():
        idx = 0
        bspp = min(args.batch_size, math.ceil(args.n_samples / accelerator.num_processes))  # batch size per process
        bs_list = amortize(args.n_samples, bspp * accelerator.num_processes)
        raw_vqvae = accelerator.unwrap_model(vqvae)
        raw_transformer = accelerator.unwrap_model(transformer)
        for bs in tqdm.tqdm(bs_list):
            fm_size = conf.data.params.img_size // 4  # feature map size
            idxmap = raw_transformer.sample(B=bspp, L=fm_size ** 2, topk=args.topk)
            z = raw_vqvae.codebook(idxmap).reshape(bspp, fm_size, fm_size, -1).permute(0, 3, 1, 2)
            samples = raw_vqvae.decode(z)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                    idx += 1

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
