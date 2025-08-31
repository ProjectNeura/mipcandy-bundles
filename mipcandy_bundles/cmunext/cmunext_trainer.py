from typing import override, Literal

from mipcandy import SegmentationTrainer, Params, LayerT, Pad2d
from torch import nn, optim

from mipcandy_bundles.cmunext.cmunext import default_norm, build_cmunext


class CMUNeXtTrainer(SegmentationTrainer):
    variant: Literal["s", "l"] | None = None

    @override
    def build_padding_module(self) -> nn.Module | None:
        return Pad2d(16)

    @override
    def build_network(self, example_shape: tuple[int, int, int] | tuple[int, int, int, int]) -> nn.Module:
        norm = default_norm if self._dataloader.batch_size > 1 else LayerT(nn.GroupNorm, num_groups=4,
                                                                           num_channels="in_ch")
        return build_cmunext(example_shape[0], self.num_classes, variant=self.variant, norm=norm)

    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.SGD(params, lr=.01, momentum=.9, weight_decay=.0001)
