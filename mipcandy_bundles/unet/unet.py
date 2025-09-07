import torch
from mipcandy import LayerT
from torch import nn
from typing import List, Type


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None, conv_op: Type[nn.Module] = nn.Conv2d,
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch", affine=True),
                 act: LayerT = LayerT(nn.ReLU, inplace=True), conv_bias: bool = True) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1: nn.Module = conv_op(in_ch, mid_ch, kernel_size=3, padding=1, bias=conv_bias)
        # self.norm1: nn.Module = norm.assemble(in_ch=mid_ch)
        self.norm1: nn.Module = norm.assemble()
        self.act1: nn.Module = act.assemble()
        self.conv2: nn.Module = conv_op(mid_ch, out_ch, kernel_size=3, padding=1, bias=conv_bias)
        # self.norm2: nn.Module = norm.assemble(in_ch=out_ch)
        self.norm2: nn.Module = norm.assemble()
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
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 2, conv_op: Type[nn.Module] = nn.Conv2d, 
                 norm_op: Type[nn.Module] = nn.InstanceNorm2d, max_pool_op: Type[nn.Module] = nn.MaxPool2d) -> None:
        super().__init__()
        self.max_pool: nn.Module = max_pool_op(kernel_size)
        self.conv: nn.Module = UNetDoubleConv(in_ch, out_ch, conv_op=conv_op, norm=LayerT(norm_op, num_features=out_ch, affine=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.max_pool(x))


class UNetUpsample(nn.Module):
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, *, conv_op: Type[nn.Module] = nn.Conv2d, norm_op: Type[nn.Module] = nn.InstanceNorm2d, bilinear: bool = True) -> None:
        super().__init__()
        if conv_op == nn.Conv3d:
            transpose_conv_op = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            transpose_conv_op = nn.ConvTranspose2d
            upsample_mode = "bilinear"
            
        if bilinear:
            self.upsample: nn.Module = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
            self.conv: nn.Module = UNetDoubleConv(up_ch + skip_ch, out_ch, conv_op=conv_op, norm=LayerT(norm_op, num_features=out_ch, affine=True))
        else:
            self.upsample: nn.Module = transpose_conv_op(up_ch, up_ch // 2, kernel_size=2, stride=2)
            self.conv: nn.Module = UNetDoubleConv(up_ch // 2 + skip_ch, out_ch, conv_op=conv_op, norm=LayerT(norm_op, num_features=out_ch, affine=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetOut(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0, conv_op: Type[nn.Module] = nn.Conv2d) -> None:
        super().__init__()
        self.conv: nn.Module = conv_op(in_ch, out_ch, kernel_size=1)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, *, bilinear: bool = False, 
                 conv_op: Type[nn.Module] = nn.Conv2d, 
                 downsample_op: LayerT = LayerT(UNetDownsample), 
                 upsample_op: LayerT = LayerT(UNetUpsample),
                 norm_op: Type[nn.Module] = nn.InstanceNorm2d, 
                 max_pool_op: Type[nn.Module] = nn.MaxPool2d, 
                 features: List[int]) -> None:
        super().__init__()
        
        self.features = features
        self.n_layers = len(features) - 1
        factor = 2 if bilinear else 1

        self.inc = UNetDoubleConv(in_ch, features[0], conv_op=conv_op, 
                                 norm=LayerT(norm_op, num_features=features[0], affine=True))
        
        self.downs = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.downs.append(
                downsample_op.assemble(features[i], features[i+1], 
                                     conv_op=conv_op, norm_op=norm_op, max_pool_op=max_pool_op)
            )
        
        self.downs.append(
            UNetDownsample(features[-2], features[-1] // factor, 
                          conv_op=conv_op, norm_op=norm_op, max_pool_op=max_pool_op)
        )
        
        self.ups = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.ups.append(
                    upsample_op.assemble(features[-1], features[-2], features[-2] // factor,
                                       conv_op=conv_op, norm_op=norm_op, bilinear=bilinear)
                )
            elif i == self.n_layers - 1:
                self.ups.append(
                    upsample_op.assemble(features[1] // factor, features[0], features[0],
                                       conv_op=conv_op, norm_op=norm_op, bilinear=bilinear)
                )
            else:
                idx = self.n_layers - 1 - i
                self.ups.append(
                    upsample_op.assemble(features[idx+1] // factor, features[idx], features[idx] // factor,
                                       conv_op=conv_op, norm_op=norm_op, bilinear=bilinear)
                )

        self.out = UNetOut(features[0], num_classes, conv_op=conv_op)

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
    model = UNet(in_ch=3, num_classes=1, conv_op=nn.Conv2d, bilinear=False, features=[32, 64, 128, 256, 512, 512, 512, 512])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sanity_check(model=model, input_shape=(1, 3, 256, 256), device=device)
    
    # 3D Sanity Check
    model = UNet(in_ch=4, num_classes=1, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d, max_pool_op=nn.MaxPool3d, bilinear=False, features=[32, 64, 128, 256, 320])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sanity_check(model=model, input_shape=(1, 4, 64, 192, 192), device=device)