import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import tqdm
from omegaconf import OmegaConf

import accelerate
import torch
from piqa import PSNR, SSIM
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.vqmodel import VQModel
from utils.data import load_data
from utils.logger import get_logger
from utils.misc import instantiate_from_config, discard_label, image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to inference configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained vqmodel weights')
    parser.add_argument('--bspp', type=int, default=256, help='Batch size on each process')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to directory saving samples (for rFID)')
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

    # BUILD DATASET & DATALOADER
    dataset = load_data(conf.data, split='test')
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.bspp,
        shuffle=False, drop_last=False, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of dataset: {len(dataset)}')
    logger.info(f'Batch size per process: {args.bspp}')
    logger.info(f'Total batch size: {args.bspp * accelerator.num_processes}')

    # BUILD MODEL
    encoder = instantiate_from_config(conf.vqmodel.encoder)
    decoder = instantiate_from_config(conf.vqmodel.decoder)
    quantizer = instantiate_from_config(conf.vqmodel.quantizer)
    vqmodel = VQModel(
        encoder=encoder, decoder=decoder, quantizer=quantizer,
        z_channels=conf.vqmodel.z_channels, codebook_dim=conf.vqmodel.codebook_dim,
    ).eval().to(device)

    # LOAD WEIGHTS
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')
    ckpt = torch.load(args.weights, map_location='cpu')
    vqmodel.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load vqvae from {args.weights}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    vqmodel, dataloader = accelerator.prepare(vqmodel, dataloader)  # type: ignore
    unwrapped_vqmodel = accelerator.unwrap_model(vqmodel)

    accelerator.wait_for_everyone()

    # START EVALUATION
    logger.info('Start evaluating...')
    idx = 0
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    psnr_fn = PSNR(reduction='none').to(device)
    ssim_fn = SSIM(reduction='none').to(device)
    psnr_list, ssim_list, indices_list = [], [], []
    with torch.no_grad():
        for x in tqdm.tqdm(dataloader, desc='Evaluating', disable=not accelerator.is_main_process):
            x = discard_label(x)
            out = vqmodel(x)
            decx, indices = out['decx'], out['indices']

            x = image_norm_to_float(x)
            decx = image_norm_to_float(decx)
            psnr = psnr_fn(decx, x)
            ssim = ssim_fn(decx, x)

            psnr = accelerator.gather_for_metrics(psnr)
            ssim = accelerator.gather_for_metrics(ssim)
            indices = accelerator.gather_for_metrics(indices)
            indices = torch.unique(indices)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            indices_list.append(indices)

            decx = accelerator.gather_for_metrics(decx)
            if args.save_dir is not None and accelerator.is_main_process:
                for img in decx:
                    save_image(img, os.path.join(args.save_dir, f'{idx}.png'))
                    idx += 1

    psnr = torch.cat(psnr_list, dim=0).mean().item()
    ssim = torch.cat(ssim_list, dim=0).mean().item()
    indices = torch.unique(torch.cat(indices_list, dim=0))
    logger.info(f'PSNR: {psnr:.4f}')
    logger.info(f'SSIM: {ssim:.4f}')
    logger.info(f'Codebook usage: {len(indices) / unwrapped_vqmodel.codebook_num * 100:.2f}%')


if __name__ == '__main__':
    main()
