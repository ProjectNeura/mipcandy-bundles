from typing import override, Mapping, Any

from mipcandy import Predictor
from torch import nn

from mipcandy_bundles.unet.unet import UNet


class UNetPredictor(Predictor):
    in_ch: int = 1
    num_classes: int = 1

    @override
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        model = UNet(self.in_ch, self.num_classes)
        model.load_state_dict(checkpoint)
        return model
