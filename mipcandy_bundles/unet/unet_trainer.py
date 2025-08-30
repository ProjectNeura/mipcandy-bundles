from typing import override

from mipcandy import SegmentationTrainer
from torch import nn

from mipcandy_bundles.unet.unet import UNet


class UNetTrainer(SegmentationTrainer):
    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return UNet(example_shape[0], self.num_classes)
