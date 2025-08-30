import torch
from mipcandy import LayerT
from torch import nn


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None,
                 norm: LayerT = LayerT(nn.BatchNorm2d, num_features="in_ch"),
                 act: LayerT = LayerT(nn.ReLU, inplace=True)) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1: nn.Module = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.norm1: nn.Module = norm.assemble(in_ch=mid_ch)
        self.act1: nn.Module = act.assemble()
        self.conv2: nn.Module = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
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
    def __init__(self, in_ch: int, out_ch: int, *, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.upsample: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv: nn.Module = UNetDoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            self.upsample: nn.Module = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv: nn.Module = UNetDoubleConv(in_ch, out_ch)

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
    def __init__(self, in_ch: int, num_classes: int, *, bilinear: bool = False) -> None:
        super().__init__()
        self.inc: nn.Module = UNetDoubleConv(in_ch, 64)
        self.down1: nn.Module = UNetDownsample(64, 128)
        self.down2: nn.Module = UNetDownsample(128, 256)
        self.down3: nn.Module = UNetDownsample(256, 512)
        factor = 2 if bilinear else 1
        self.down4: nn.Module = UNetDownsample(512, 1024 // factor)
        self.up1: nn.Module = UNetUpsample(1024, 512 // factor, bilinear=bilinear)
        self.up2: nn.Module = UNetUpsample(512, 256 // factor, bilinear=bilinear)
        self.up3: nn.Module = UNetUpsample(256, 128 // factor, bilinear=bilinear)
        self.up4: nn.Module = UNetUpsample(128, 64, bilinear=bilinear)
        self.out: nn.Module = UNetOut(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
