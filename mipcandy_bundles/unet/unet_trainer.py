from typing import override

from mipcandy import SegmentationTrainer
from torch import nn

from mipcandy_bundles.unet.unet import make_unet2d, make_unet3d


class UNetTrainer(SegmentationTrainer):
    num_dims: int = 2

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return make_unet2d(example_shape[0], self.num_classes) if self.num_dims == 2 else make_unet3d(example_shape[0],
                                                                                                      self.num_classes)
