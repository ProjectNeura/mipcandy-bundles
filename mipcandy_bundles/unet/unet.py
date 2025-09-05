import torch
from mipcandy import LayerT
from torch import nn
from typing import List


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None,
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch", affine=True),
                 act: LayerT = LayerT(nn.ReLU, inplace=True), conv_bias: bool = True) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1: nn.Module = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=conv_bias)
        self.norm1: nn.Module = norm.assemble(in_ch=mid_ch)
        self.act1: nn.Module = act.assemble()
        self.conv2: nn.Module = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=conv_bias)
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
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 2) -> None:
        super().__init__()
        self.max_pool: nn.Module = nn.MaxPool2d(kernel_size)
        self.conv: nn.Module = UNetDoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.max_pool(x))


class UNetUpsample(nn.Module):
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, *, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.upsample: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv: nn.Module = UNetDoubleConv(up_ch + skip_ch, out_ch)
        else:
            self.upsample: nn.Module = nn.ConvTranspose2d(up_ch, up_ch // 2, kernel_size=2, stride=2)
            self.conv: nn.Module = UNetDoubleConv(up_ch // 2 + skip_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetOut(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0) -> None:
        super().__init__()
        self.conv: nn.Module = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, *, bilinear: bool = False, downsample_op: LayerT = LayerT(UNetDownsample), upsample_op: LayerT = LayerT(UNetUpsample),
                 features: List[int]) -> None:
        super().__init__()
        
        self.features = features
        
        self.inc: nn.Module = UNetDoubleConv(in_ch, features[0])
        self.down1: nn.Module = downsample_op.assemble(features[0], features[1])
        self.down2: nn.Module = downsample_op.assemble(features[1], features[2])
        self.down3: nn.Module = downsample_op.assemble(features[2], features[3])
        self.down4: nn.Module = downsample_op.assemble(features[3], features[4])
        self.down5: nn.Module = downsample_op.assemble(features[4], features[5])
        self.down6: nn.Module = downsample_op.assemble(features[5], features[6])
        factor = 2 if bilinear else 1
        self.down7: nn.Module = UNetDownsample(features[6], features[7] // factor)
        
        self.up1: nn.Module = upsample_op.assemble(features[7], features[6], features[6] // factor, bilinear=bilinear)
        self.up2: nn.Module = upsample_op.assemble(features[6] // factor, features[5], features[5] // factor, bilinear=bilinear)
        self.up3: nn.Module = upsample_op.assemble(features[5] // factor, features[4], features[4] // factor, bilinear=bilinear)
        self.up4: nn.Module = upsample_op.assemble(features[4] // factor, features[3], features[3] // factor, bilinear=bilinear)
        self.up5: nn.Module = upsample_op.assemble(features[3] // factor, features[2], features[2] // factor, bilinear=bilinear)
        self.up6: nn.Module = upsample_op.assemble(features[2] // factor, features[1], features[1] // factor, bilinear=bilinear)
        self.up7: nn.Module = upsample_op.assemble(features[1] // factor, features[0], features[0], bilinear=bilinear)
        
        self.out: nn.Module = UNetOut(features[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        
        return self.out(x)

if __name__ == "__main__":
    from mipcandy import sanity_check

    model = UNet(in_ch=3, num_classes=1, bilinear=False, features=[32, 64, 128, 256, 512, 512, 512, 512])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sanity_check(model=model, input_shape=(1, 3, 256, 256), device=device)