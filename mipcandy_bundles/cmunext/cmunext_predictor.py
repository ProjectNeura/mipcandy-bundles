from typing import override, Mapping, Any, Literal

from mipcandy import Predictor
from torch import nn

from mipcandy import Pad2d, LayerT
from mipcandy_bundles.cmunext.cmunext import default_norm, build_cmunext


class CMUNeXtPredictor(Predictor):
    variant: Literal["s", "l"] | None = None
    in_ch: int = 1
    num_classes: int = 1
    norm: LayerT = default_norm

    @override
    def build_padding_module(self) -> nn.Module | None:
        return Pad2d(16)

    @override
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        model = build_cmunext(self.in_ch, 1, variant=self.variant, norm=self.norm)
        model.load_state_dict(checkpoint)
        return model
