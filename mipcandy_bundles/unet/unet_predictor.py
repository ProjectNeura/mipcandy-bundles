from typing import override, Mapping, Any, Literal

from mipcandy import Predictor
from torch import nn

from mipcandy_bundles.unet.unet import make_unet2d, make_unet3d


class UNetPredictor(Predictor):
    in_ch: int = 1
    num_classes: int = 1
    num_dims: Literal[2, 3] = 2

    @override
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        model = make_unet2d(self.in_ch, self.num_classes) if self.num_dims == 2 else make_unet3d(self.in_ch,
                                                                                                 self.num_classes)
        model.load_state_dict(checkpoint)
        return model
