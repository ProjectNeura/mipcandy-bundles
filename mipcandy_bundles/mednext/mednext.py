import torch
import torch.nn as nn
from mipcandy import LayerT
from typing import Literal, Sequence, override

class MedNeXtBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, exp_r: int = 4, kernel_size: int = 7, residual: bool = True,
                 num_dims: Literal[2, 3] = 3, conv: LayerT = LayerT(nn.Conv3d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch"),
                 act: LayerT = LayerT(nn.GELU)) -> None:
        super().__init__()
        self.residual: bool = residual
        mid_ch = exp_r * in_ch
        self.conv1: nn.Module = conv.assemble(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_ch)
        self.norm: nn.Module = norm.update(num_groups=in_ch).assemble(in_ch=in_ch)
        self.conv2: nn.Module = conv.assemble(in_ch, mid_ch, kernel_size=1)
        self.act: nn.Module = act.assemble()
        self.conv3: nn.Module = conv.assemble(mid_ch, out_ch, kernel_size=1)
        spatial_dims = (1,) * num_dims
        self.grn_beta = nn.Parameter(torch.zeros(1, mid_ch, *spatial_dims))
        self.grn_gamma = nn.Parameter(torch.zeros(1, mid_ch, *spatial_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        spatial_dims = tuple(range(2, x.ndim))
        gx = torch.norm(x, p=2, dim=spatial_dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.grn_gamma * (x * nx) + self.grn_beta + x
        x = self.conv3(x)
        if self.residual:
            x = x + residual
        return x


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(self, in_ch: int, out_ch: int, *, exp_r: int = 4, kernel_size: int = 7, residual: bool = False,
                 num_dims: Literal[2, 3] = 3, conv: LayerT = LayerT(nn.Conv3d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch"),
                 act: LayerT = LayerT(nn.GELU)) -> None:
        super().__init__(in_ch, out_ch, exp_r=exp_r, kernel_size=kernel_size, residual=False,
                        num_dims=num_dims, conv=conv, norm=norm, act=act)
        self.conv1: nn.Module = conv.assemble(in_ch, in_ch, kernel_size=kernel_size, stride=2,
                                             padding=kernel_size // 2, groups=in_ch)
        self.resample_residual: bool = residual
        if residual:
            self.res_conv: nn.Module = conv.assemble(in_ch, out_ch, kernel_size=1, stride=2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = super().forward(x)
        if self.resample_residual:
            x = x + self.res_conv(identity)
        return x


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, in_ch: int, out_ch: int, *, exp_r: int = 4, kernel_size: int = 7, residual: bool = False,
                 num_dims: Literal[2, 3] = 3, conv: LayerT = LayerT(nn.Conv3d),
                 transpose_conv: LayerT = LayerT(nn.ConvTranspose3d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch"),
                 act: LayerT = LayerT(nn.GELU)) -> None:
        super().__init__(in_ch, out_ch, exp_r=exp_r, kernel_size=kernel_size, residual=False,
                        num_dims=num_dims, conv=conv, norm=norm, act=act)
        self.conv1: nn.Module = transpose_conv.assemble(in_ch, in_ch, kernel_size=kernel_size, stride=2,
                                                       padding=kernel_size // 2, groups=in_ch)
        self.num_dims: int = num_dims
        self.resample_residual: bool = residual
        if residual:
            self.res_conv: nn.Module = transpose_conv.assemble(in_ch, out_ch, kernel_size=1, stride=2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = super().forward(x)
        padding = (1, 0) * self.num_dims
        x = nn.functional.pad(x, padding)
        if self.resample_residual:
            res = self.res_conv(identity)
            res = nn.functional.pad(res, padding)
            x = x + res
        return x


class MedNeXtOut(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0,
                 transpose_conv: LayerT = LayerT(nn.ConvTranspose3d)) -> None:
        super().__init__()
        self.conv: nn.Module = transpose_conv.assemble(in_ch, out_ch, kernel_size=1)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class MedNeXt(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden_chs: Sequence[int], *, num_dims: Literal[2, 3] = 3,
                 exp_r: int | Sequence[int] = 4, kernel_size: int = 7, deep_supervision: bool = False,
                 dropout: float = 0, block_counts: Sequence[int] | None = None, residual: bool = True,
                 resample_residual: bool = False, conv: LayerT = LayerT(nn.Conv3d),
                 transpose_conv: LayerT = LayerT(nn.ConvTranspose3d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch"),
                 act: LayerT = LayerT(nn.GELU)) -> None:
        super().__init__()
        self.num_layers: int = len(hidden_chs)
        self.deep_supervision: bool = deep_supervision
        if block_counts is None:
            block_counts = [2] * (2 * self.num_layers - 1)
        if len(block_counts) != 2 * self.num_layers - 1:
            raise ValueError(f"block_counts must have {2 * self.num_layers - 1} elements for {self.num_layers} layers, got {len(block_counts)}")
        if isinstance(exp_r, int):
            exp_r = [exp_r] * len(block_counts)
        self.stem: nn.Module = conv.assemble(in_ch, hidden_chs[0], kernel_size=1)
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.downs: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_layers - 1):
            enc_block = nn.Sequential(*[
                MedNeXtBlock(hidden_chs[i], hidden_chs[i], exp_r=exp_r[i], kernel_size=kernel_size,
                           residual=residual, num_dims=num_dims, conv=conv, norm=norm, act=act)
                for _ in range(block_counts[i])
            ])
            self.enc_blocks.append(enc_block)
            down = MedNeXtDownBlock(hidden_chs[i], hidden_chs[i + 1], exp_r=exp_r[i + 1],
                                   kernel_size=kernel_size, residual=resample_residual, num_dims=num_dims,
                                   conv=conv, norm=norm, act=act)
            self.downs.append(down)
        bottleneck_idx: int = self.num_layers - 1
        self.bottleneck: nn.Sequential = nn.Sequential(*[
            MedNeXtBlock(hidden_chs[-1], hidden_chs[-1], exp_r=exp_r[bottleneck_idx], kernel_size=kernel_size,
                       residual=residual, num_dims=num_dims, conv=conv, norm=norm, act=act)
            for _ in range(block_counts[bottleneck_idx])
        ])
        self.ups: nn.ModuleList = nn.ModuleList()
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dec_idx = self.num_layers + i
            layer_idx = self.num_layers - 2 - i
            up = MedNeXtUpBlock(hidden_chs[layer_idx + 1], hidden_chs[layer_idx], exp_r=exp_r[dec_idx],
                              kernel_size=kernel_size, residual=resample_residual, num_dims=num_dims, conv=conv,
                              transpose_conv=transpose_conv, norm=norm, act=act)
            self.ups.append(up)
            dec_block = nn.Sequential(*[
                MedNeXtBlock(hidden_chs[layer_idx], hidden_chs[layer_idx], exp_r=exp_r[dec_idx],
                           kernel_size=kernel_size, residual=residual, num_dims=num_dims, conv=conv, norm=norm, act=act)
                for _ in range(block_counts[dec_idx])
            ])
            self.dec_blocks.append(dec_block)
        self.out: nn.Module = MedNeXtOut(hidden_chs[0], num_classes, dropout=dropout, transpose_conv=transpose_conv)
        if self.deep_supervision:
            self.out_layers: nn.ModuleList = nn.ModuleList([
                MedNeXtOut(hidden_chs[i], num_classes, dropout=dropout, transpose_conv=transpose_conv)
                for i in range(self.num_layers)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x = self.stem(x)
        skip_connections = []
        for enc_block, down in zip(self.enc_blocks, self.downs):
            x = enc_block(x)
            skip_connections.append(x)
            x = down(x)
        x = self.bottleneck(x)
        if self.deep_supervision and self.training:
            ds_outputs = []
            ds_outputs.append(self.out_layers[-1](x))
        for i, (up, dec_block) in enumerate(zip(self.ups, self.dec_blocks)):
            skip_idx = len(skip_connections) - 1 - i
            x = up(x)
            x = x + skip_connections[skip_idx]
            x = dec_block(x)
            if self.deep_supervision and self.training and skip_idx > 0:
                ds_outputs.append(self.out_layers[skip_idx](x))
        x = self.out(x)
        if self.deep_supervision and self.training:
            return ds_outputs[::-1] + [x]
        else:
            return x


def make_mednext2d(in_ch: int, num_classes: int, *, hidden_chs: Sequence[int] = (32, 64, 128, 256, 512),
                   exp_r: int | Sequence[int] = 4, kernel_size: int = 7, deep_supervision: bool = False,
                   dropout: float = 0, block_counts: Sequence[int] | None = None, residual: bool = True,
                   resample_residual: bool = False) -> MedNeXt:
    return MedNeXt(in_ch, num_classes, hidden_chs=hidden_chs, num_dims=2, exp_r=exp_r, kernel_size=kernel_size,
                   deep_supervision=deep_supervision, dropout=dropout, block_counts=block_counts,
                   residual=residual, resample_residual=resample_residual,
                   conv=LayerT(nn.Conv2d), transpose_conv=LayerT(nn.ConvTranspose2d),
                   norm=LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch"))


def make_mednext3d(in_ch: int, num_classes: int, *, hidden_chs: Sequence[int] = (32, 64, 128, 256, 512),
                   exp_r: int | Sequence[int] = 4, kernel_size: int = 7, deep_supervision: bool = False,
                   dropout: float = 0, block_counts: Sequence[int] | None = None, residual: bool = True,
                   resample_residual: bool = False) -> MedNeXt:
    return MedNeXt(in_ch, num_classes, hidden_chs=hidden_chs, num_dims=3, exp_r=exp_r, kernel_size=kernel_size,
                   deep_supervision=deep_supervision, dropout=dropout, block_counts=block_counts,
                   residual=residual, resample_residual=resample_residual,
                   conv=LayerT(nn.Conv3d), transpose_conv=LayerT(nn.ConvTranspose3d),
                   norm=LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch"))


if __name__ == "__main__":
    from mipcandy import sanity_check

    print("Testing 2D Model (without deep supervision)")
    model_2d = make_mednext2d(1, 13)
    result_2d = sanity_check(model_2d, (1, 128, 128))
    print(result_2d.layer_stats)
    print(result_2d)
    print(f"Output shape: {result_2d.output.shape}")

    print("Testing 3D Model (without deep supervision)")
    model_3d = make_mednext3d(1, 13)
    result_3d = sanity_check(model_3d, (1, 64, 64, 64))
    print(result_3d.layer_stats)
    print(result_3d)
    print(f"Output shape: {result_3d.output.shape}")