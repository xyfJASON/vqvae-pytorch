import torch
import torch.nn as nn
from torch import Tensor

from models.net import Encoder, Decoder
from models.quantizer import VectorQuantizer, VectorQuantizerEMA


class VQVAE(nn.Module):
    def __init__(
            self,
            img_channels: int = 3,
            hidden_dim: int = 256,
            n_resblocks: int = 2,
            codebook_num: int = 512,
            codebook_dim: int = 64,
            codebook_update: str = 'learned',
            ema_decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = Encoder(img_channels, hidden_dim, n_resblocks, codebook_dim)
        self.decoder = Decoder(img_channels, hidden_dim, n_resblocks, codebook_dim)
        if codebook_update == 'learned':
            self.quantizer = VectorQuantizer(codebook_num, codebook_dim)
        elif codebook_update == 'kmeans':
            self.quantizer = VectorQuantizerEMA(codebook_num, codebook_dim, ema_decay)
        else:
            raise ValueError(f'Codebook update method {codebook_update} is not supported')

    @property
    def codebook(self):
        return self.quantizer.codebook

    def update_codebook(self, new_sumz, new_sumn):
        if isinstance(self.quantizer, VectorQuantizerEMA):
            self.quantizer.update_codebook(new_sumz, new_sumn)

    def forward(self, x: Tensor, return_dict: bool = True):
        z = self.encoder(x)
        quantized_z, quantized_z_st, indices, perplexity = self.quantizer(z)
        decx = self.decoder(quantized_z_st)

        if return_dict:
            return dict(
                decx=decx,
                z=z,
                quantized_z=quantized_z,
                quantized_z_st=quantized_z_st,
                indices=indices,
                perplexity=perplexity,
            )
        else:
            return decx, z, quantized_z, quantized_z_st, indices, perplexity

    def get_latents(self, x: Tensor, return_dict: bool = True):
        z = self.encoder(x)
        quantized_z, quantized_z_st, indices, perplexity = self.quantizer(z)

        if return_dict:
            return dict(
                quantized_z=quantized_z,
                quantized_z_st=quantized_z_st,
                indices=indices,
                perplexity=perplexity,
            )
        else:
            return quantized_z, quantized_z_st, indices, perplexity

    def decode(self, z: Tensor):
        return self.decoder(z)

    def reconstruct(self, x: Tensor):
        return self.forward(x)['decx']


def _test():
    vqvae = VQVAE()
    print(sum(p.numel() for p in vqvae.parameters()))

    x = torch.rand((10, 3, 64, 64))
    out = vqvae(x)
    print('decx\t\t', out['decx'].shape)
    print('quantized_z\t', out['quantized_z'].shape)
    print('quantized_z_st\t', out['quantized_z_st'].shape)
    print('indices\t\t', out['indices'].shape)
    print('perplexity\t', out['perplexity'])


if __name__ == '__main__':
    _test()
