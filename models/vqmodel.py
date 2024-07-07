import torch.nn as nn
from torch import Tensor

from .quantizer import VectorQuantizer


class VQModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            quantizer: VectorQuantizer,
            z_channels: int,
            codebook_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.enc2code = nn.Conv2d(z_channels, codebook_dim, 1)
        self.code2dec = nn.Conv2d(codebook_dim, z_channels, 1)

    @property
    def codebook(self):
        return self.quantizer.codebook

    @property
    def codebook_dim(self):
        return self.quantizer.codebook_dim

    @property
    def codebook_num(self):
        return self.quantizer.codebook_num

    def forward(self, x: Tensor):
        z = self.encoder(x)
        z = self.enc2code(z)
        quantized_z, quantized_z_st, indices, perplexity = self.quantizer(z)
        decx = self.code2dec(quantized_z_st)
        decx = self.decoder(decx)
        return dict(
            decx=decx, z=z, indices=indices, perplexity=perplexity,
            quantized_z=quantized_z, quantized_z_st=quantized_z_st,
        )

    def encode(self, x: Tensor):
        z = self.encoder(x)
        z = self.enc2code(z)
        quantized_z, quantized_z_st, indices, perplexity = self.quantizer(z)
        return dict(
            z=z, indices=indices, perplexity=perplexity,
            quantized_z=quantized_z, quantized_z_st=quantized_z_st,
        )

    def decode(self, z: Tensor):
        decx = self.code2dec(z)
        decx = self.decoder(decx)
        return decx
