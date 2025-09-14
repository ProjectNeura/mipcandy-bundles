import torch
from mipcandy import LayerT
from torch import nn
from typing import Literal, Sequence


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch", affine=True),
                 act: LayerT = LayerT(nn.ReLU, inplace=True), bias: bool = True) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1: nn.Module = conv.assemble(in_ch, mid_ch, kernel_size=3, padding=1, bias=bias)
        self.norm1: nn.Module = norm.assemble(in_ch=mid_ch)
        self.act1: nn.Module = act.assemble()
        self.conv2: nn.Module = conv.assemble(mid_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.norm2: nn.Module = norm.assemble(in_ch=out_ch)
        self.act2: nn.Module = act.assemble()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class UNetDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 2, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch"),
                 max_pool: LayerT = LayerT(nn.MaxPool2d)) -> None:
        super().__init__()
        self.max_pool: nn.Module = max_pool.assemble(kernel_size)
        self.conv: nn.Module = UNetDoubleConv(in_ch, out_ch, conv=conv, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.max_pool(x))


class UNetUpsample(nn.Module):
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, *, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch"), linear: bool = True,
                 num_dims: Literal[2, 3] = 2) -> None:
        super().__init__()
        if num_dims == 2:
            transpose_conv = LayerT(nn.ConvTranspose2d)
            upsample_mode = "bilinear"
        else:
            transpose_conv = LayerT(nn.ConvTranspose3d)
            upsample_mode = "trilinear"
        if linear:
            self.upsample: nn.Module = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
            self.conv: nn.Module = UNetDoubleConv(up_ch + skip_ch, out_ch, conv=conv, norm=norm)
        else:
            self.upsample: nn.Module = transpose_conv.assemble(up_ch, up_ch // 2, kernel_size=2, stride=2)
            self.conv: nn.Module = UNetDoubleConv(up_ch // 2 + skip_ch, out_ch, conv=conv, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetOut(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0, conv: LayerT = LayerT(nn.Conv2d)) -> None:
        super().__init__()
        self.conv: nn.Module = conv.assemble(in_ch, out_ch, kernel_size=1)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden_chs: Sequence[int], *, num_dims: Literal[2, 3] = 2,
                 linear: bool = False, conv: LayerT = LayerT(nn.Conv2d), downsample: LayerT = LayerT(UNetDownsample),
                 upsample: LayerT = LayerT(UNetUpsample), norm: LayerT = LayerT(nn.InstanceNorm2d),
                 max_pool: LayerT = LayerT(nn.MaxPool2d)) -> None:
        super().__init__()
        self.hidden_chs: Sequence[int] = hidden_chs
        self.num_layers = len(hidden_chs) - 1
        factor = 2 if linear else 1
        self.inc: nn.Module = UNetDoubleConv(in_ch, hidden_chs[0], conv=conv, norm=norm)
        self.downs: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.downs.append(downsample.assemble(
                hidden_chs[i], hidden_chs[i + 1], conv=conv, norm=norm, max_pool=max_pool
            ))
        self.downs.append(UNetDownsample(
            hidden_chs[-2], hidden_chs[-1] // factor, conv=conv, norm=norm, max_pool=max_pool
        ))
        self.ups: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.ups.append(upsample.assemble(
                    hidden_chs[-1], hidden_chs[-2], hidden_chs[-2] // factor, conv=conv, norm=norm, linear=linear,
                    num_dims=num_dims
                ))
            elif i == self.num_layers - 1:
                self.ups.append(upsample.assemble(
                    hidden_chs[1] // factor, hidden_chs[0], hidden_chs[0], conv=conv, norm=norm, linear=linear,
                    num_dims=num_dims
                ))
            else:
                idx = self.num_layers - 1 - i
                self.ups.append(upsample.assemble(
                    hidden_chs[idx + 1] // factor, hidden_chs[idx], hidden_chs[idx] // factor, conv=conv, norm=norm,
                    linear=linear, num_dims=num_dims
                ))
        self.out: nn.Module = UNetOut(hidden_chs[0], num_classes, conv=conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = []
        x = self.inc(x)
        cache.append(x)
        for down in self.downs[:-1]:
            x = down(x)
            cache.append(x)
        x = self.downs[-1](x)
        for i, up in enumerate(self.ups):
            skip_idx = len(cache) - 1 - i
            x = up(x, cache[skip_idx])
        return self.out(x)


def make_unet2d(in_ch: int, num_classes: int, *, hidden_chs: Sequence[int] = (32, 64, 128, 256, 512, 512, 512, 512),
                linear: bool = False) -> UNet:
    return UNet(in_ch, num_classes, hidden_chs, linear=linear)


def make_unet3d(in_ch: int, num_classes: int, *, hidden_chs: Sequence[int] = (32, 64, 128, 256, 320),
                linear: bool = False) -> UNet:
    return UNet(in_ch, num_classes, hidden_chs, num_dims=3, linear=linear, conv=LayerT(nn.Conv3d),
                norm=LayerT(nn.InstanceNorm3d, num_features="in_ch"), max_pool=LayerT(nn.MaxPool3d))


if __name__ == "__main__":
    from mipcandy import sanity_check

    model = make_unet2d(3, 1)
    result_2d = sanity_check(model, (3, 256, 256))
    print(result_2d.layer_stats)
    print(result_2d)
    print(result_2d.output.shape)

    model = make_unet3d(4, 1)
    result_3d = sanity_check(model, (4, 64, 192, 192))
    print(result_3d.layer_stats)
    print(result_3d)
    print(result_3d.output.shape)
