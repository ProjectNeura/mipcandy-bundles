from typing import override

from mipcandy import SegmentationTrainer
from torch import nn

from mipcandy_bundles.nestedunet.unetpp import make_unetpp2d


class UNetPPTrainer(SegmentationTrainer):
    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return make_unetpp2d(example_shape[0], self.num_classes)
