from typing import Literal

import torch
from mipcandy import LayerT, ConvBlock2d
from torch import nn

default_norm: LayerT = LayerT(nn.BatchNorm2d, num_features="in_ch")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn: nn.Module = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 3, fusion: bool = False,
                 norm: LayerT = default_norm) -> None:
        super().__init__()
        if fusion:
            self.conv1: nn.Module = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=2)
        else:
            self.conv1: nn.Module = nn.Conv2d(in_ch, in_ch, kernel_size, groups=in_ch, padding=kernel_size // 2)
        self.act1: nn.Module = nn.GELU()
        self.norm1: nn.Module = norm.assemble(in_ch=in_ch)
        self.conv2: nn.Module = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.act2: nn.Module = nn.GELU()
        self.norm2: nn.Module = norm.assemble(in_ch=out_ch * 4)
        self.conv3: nn.Module = nn.Conv2d(out_ch * 4, out_ch, 1)
        self.act3: nn.Module = nn.GELU()
        self.norm3: nn.Module = norm.assemble(in_ch=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.norm3(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int = 1, kernel_size: int = 3,
                 norm: LayerT = default_norm) -> None:
        super().__init__()
        self.blocks: nn.ModuleList[nn.Module] = nn.Sequential(*(
            CMUNeXtBlock(in_ch, in_ch, kernel_size=kernel_size, norm=norm) for _ in range(depth)
        ))
        self.up: nn.Module = ConvBlock2d(in_ch, out_ch, 3, padding=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.up(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm: LayerT = default_norm) -> None:
        super().__init__()
        self.upsample: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv: nn.Module = ConvBlock2d(in_ch, out_ch, 3, padding=1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm: LayerT = default_norm) -> None:
        super().__init__()
        self.up: nn.Module = Upsample(in_ch, out_ch)
        self.fusion: nn.Module = CMUNeXtBlock(out_ch * 2, out_ch, fusion=True, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.fusion(x)
        return x


class CMUNeXt(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, *, dims: tuple[int, int, int, int, int] = (16, 32, 128, 160, 256),
                 depths: tuple[int, int, int, int, int] = (1, 1, 1, 3, 1),
                 kernels: tuple[int, int, int, int, int] = (3, 3, 7, 7, 7), norm: LayerT = default_norm) -> None:
        super().__init__()
        self.max_pool: nn.Module = nn.MaxPool2d(2, 2)
        self.stem: nn.Module = ConvBlock2d(in_ch, dims[0], 3, padding=1, norm=norm)
        self.encoder1: nn.Module = Encoder(dims[0], dims[0], depth=depths[0], kernel_size=kernels[0], norm=norm)
        self.encoder2: nn.Module = Encoder(dims[0], dims[1], depth=depths[1], kernel_size=kernels[1], norm=norm)
        self.encoder3: nn.Module = Encoder(dims[1], dims[2], depth=depths[2], kernel_size=kernels[2], norm=norm)
        self.encoder4: nn.Module = Encoder(dims[2], dims[3], depth=depths[3], kernel_size=kernels[3], norm=norm)
        self.encoder5: nn.Module = Encoder(dims[3], dims[4], depth=depths[4], kernel_size=kernels[4], norm=norm)
        self.decoder1: nn.Module = Decoder(dims[4], dims[3], norm=norm)
        self.decoder2: nn.Module = Decoder(dims[3], dims[2], norm=norm)
        self.decoder3: nn.Module = Decoder(dims[2], dims[1], norm=norm)
        self.decoder4: nn.Module = Decoder(dims[1], dims[0], norm=norm)
        self.out: nn.Module = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x1 = self.encoder1(x)
        x2 = self.max_pool(x1)
        x2 = self.encoder2(x2)
        x3 = self.max_pool(x2)
        x3 = self.encoder3(x3)
        x4 = self.max_pool(x3)
        x4 = self.encoder4(x4)
        x5 = self.max_pool(x4)
        x5 = self.encoder5(x5)
        d1 = self.decoder1(x5, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)
        x = self.out(d4)
        return x


def build_cmunext(in_ch: int, num_classes: int, *, variant: Literal["s", "l"] | None = None,
                  norm: LayerT = default_norm) -> CMUNeXt:
    match variant:
        case "s":
            return CMUNeXt(in_ch, num_classes, dims=(8, 16, 32, 64, 128), depths=(1, 1, 1, 1, 1),
                           kernels=(3, 3, 7, 7, 9), norm=norm)
        case "l":
            return CMUNeXt(in_ch, num_classes, dims=(32, 64, 128, 256, 512), depths=(1, 1, 1, 6, 3),
                           kernels=(3, 3, 7, 7, 7), norm=norm)
        case None:
            return CMUNeXt(in_ch, num_classes, norm=norm)
