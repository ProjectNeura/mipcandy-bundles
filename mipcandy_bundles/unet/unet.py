import torch
from mipcandy import LayerT
from torch import nn
from typing import Literal


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.InstanceNorm2d, affine=True),
                 act: LayerT = LayerT(nn.ReLU, inplace=True), bias: bool = True) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1: nn.Module = conv.assemble(in_ch, mid_ch, kernel_size=3, padding=1, bias=bias)
        self.norm1: nn.Module = norm.assemble(num_features=mid_ch)
        self.act1: nn.Module = act.assemble()
        self.conv2: nn.Module = conv.assemble(mid_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.norm2: nn.Module = norm.assemble(num_features=out_ch)
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
                 norm: LayerT = LayerT(nn.InstanceNorm2d), max_pool: LayerT = LayerT(nn.MaxPool2d)) -> None:
        super().__init__()
        self.max_pool: nn.Module = max_pool.assemble(kernel_size)
        self.conv: nn.Module = UNetDoubleConv(in_ch, out_ch, conv=conv, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.max_pool(x))


class UNetUpsample(nn.Module):
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, *,
                 conv: LayerT = LayerT(nn.Conv2d), norm: LayerT = LayerT(nn.InstanceNorm2d),
                 linear: bool = True, num_dims: Literal[2, 3]) -> None:
        super().__init__()
        if num_dims == 2:
            transpose_conv = LayerT(nn.ConvTranspose2d)
            upsample_mode = "bilinear"
        elif num_dims == 3:
            transpose_conv = LayerT(nn.ConvTranspose3d)
            upsample_mode = "trilinear"
        else:
            raise ValueError("num_dims must be 2 or 3")

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
    def __init__(self, in_ch: int, num_classes: int, *, linear: bool = False, 
                 conv: LayerT = LayerT(nn.Conv2d), 
                 downsample: LayerT = LayerT(UNetDownsample), 
                 upsample: LayerT = LayerT(UNetUpsample),
                 norm: LayerT = LayerT(nn.InstanceNorm2d), 
                 max_pool: LayerT = LayerT(nn.MaxPool2d), 
                 features: list[int]) -> None:
        super().__init__()

        self.features = features
        self.n_layers = len(features) - 1
        factor = 2 if linear else 1

        self.inc: nn.Module = UNetDoubleConv(in_ch, features[0], conv=conv, norm=norm)
        
        self.downs: nn.ModuleList = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.downs.append(
                downsample.assemble(features[i], features[i+1], 
                                     conv=conv, norm=norm, max_pool=max_pool)
            )
        
        self.downs.append(
            UNetDownsample(features[-2], features[-1] // factor, 
                          conv=conv, norm=norm, max_pool=max_pool)
        )
        
        self.ups: nn.ModuleList = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.ups.append(
                    upsample.assemble(features[-1], features[-2], features[-2] // factor,
                                       conv=conv, norm=norm, linear=linear, num_dims=2 if conv.m == nn.Conv2d else 3)
                )
            elif i == self.n_layers - 1:
                self.ups.append(
                    upsample.assemble(features[1] // factor, features[0], features[0],
                                       conv=conv, norm=norm, linear=linear, num_dims=2 if conv.m == nn.Conv2d else 3)
                )
            else:
                idx = self.n_layers - 1 - i
                self.ups.append(
                    upsample.assemble(features[idx+1] // factor, features[idx], features[idx] // factor,
                                       conv=conv, norm=norm, linear=linear, num_dims=2 if conv.m == nn.Conv2d else 3)
                )

        self.out = UNetOut(features[0], num_classes, conv=conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_features = []
        
        x = self.inc(x)
        skip_features.append(x)
        
        for down in self.downs[:-1]:
            x = down(x)
            skip_features.append(x)
        
        x = self.downs[-1](x)
        
        for i, up in enumerate(self.ups):
            skip_idx = len(skip_features) - 1 - i
            x = up(x, skip_features[skip_idx])
        
        return self.out(x)

if __name__ == "__main__":
    from mipcandy import sanity_check

    # 2D Sanity Check
    model = UNet(in_ch=3, num_classes=1, conv=LayerT(nn.Conv2d), linear=False, features=[32, 64, 128, 256, 512, 512, 512, 512])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result_2d = sanity_check(model=model, input_shape=(3, 256, 256), device=device)
    print(result_2d.layer_stats)
    print(result_2d)
    print(result_2d.output.shape)

    # 3D Sanity Check
    model = UNet(in_ch=4, num_classes=1, conv=LayerT(nn.Conv3d), norm=LayerT(nn.InstanceNorm3d), max_pool=LayerT(nn.MaxPool3d), linear=False, features=[32, 64, 128, 256, 320])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result_3d = sanity_check(model=model, input_shape=(4, 64, 192, 192), device=device)
    print(result_3d.layer_stats)
    print(result_3d)
    print(result_3d.output.shape)