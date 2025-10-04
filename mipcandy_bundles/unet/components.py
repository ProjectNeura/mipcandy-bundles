import torch
from mipcandy import LayerT
from torch import nn
from typing import Literal
from mipcandy_bundles.unet.unet import UNetUpsample, UNetDoubleConv


class AttentionGate(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int, *,
                 num_dims: Literal[2, 3] = 2,
                 conv: LayerT | None = None,
                 norm: LayerT | None = None) -> None:
        super().__init__()

        if conv is None:
            conv = LayerT(nn.Conv3d) if num_dims == 3 else LayerT(nn.Conv2d)
        if norm is None:
            norm = LayerT(nn.InstanceNorm3d, num_features="in_ch", affine=True) if num_dims == 3 else LayerT(nn.InstanceNorm2d, affine=True)
        self.W_g: nn.Module = nn.Sequential(
            conv.assemble(gate_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            norm.assemble(in_ch=inter_ch)
        )
        self.W_x: nn.Module = nn.Sequential(
            conv.assemble(skip_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            norm.assemble(in_ch=inter_ch)
        )
        self.psi: nn.Module = nn.Sequential(
            conv.assemble(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu: nn.Module = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip * psi


class UNetResidualConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, mid_ch: int | None = None,
                 conv: LayerT = LayerT(nn.Conv2d),
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
        self.shortcut: nn.Module
        if in_ch != out_ch:
            self.shortcut = conv.assemble(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act2(x)
        return x


class UNetAttentionUpsample(UNetUpsample):
    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, *,
                 conv: LayerT = LayerT(nn.Conv2d), 
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch", affine=True),
                 linear: bool = True, num_dims: Literal[2, 3],
                 inter_ch_ratio: int = 2, conv_block: LayerT = LayerT(UNetDoubleConv)) -> None:
        super().__init__(up_ch, skip_ch, out_ch, conv=conv, norm=norm, linear=linear, 
                         num_dims=num_dims, conv_block=conv_block)

        eff_up_ch = up_ch if linear else (up_ch // 2)
        expected_in_ch = eff_up_ch + skip_ch
        first_conv = getattr(self.conv, 'conv1', None)
        if first_conv is not None and hasattr(first_conv, 'in_channels'):
            assert first_conv.in_channels == expected_in_ch, f"Channel mismatch: expected {expected_in_ch}, got {first_conv.in_channels}"
        inter_ch = max(1, skip_ch // inter_ch_ratio)
        self.attention_gate: nn.Module = AttentionGate(eff_up_ch, skip_ch, inter_ch,
                                                      num_dims=num_dims, conv=conv, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        x2_att = self.attention_gate(x1, x2)
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)