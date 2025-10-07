from typing import override

from mipcandy import SegmentationTrainer
import torch
from torch import nn
import torch.nn.functional as F
from mipcandy import TrainerToolbox

from mipcandy_bundles.mednext.mednext import make_mednext2d, make_mednext3d


class MedNeXtTrainer(SegmentationTrainer):
    deep_supervision: bool = False

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        if self.deep_supervision:
            self.log("Enabled deep supervision.")
        return make_mednext2d(example_shape[0], self.num_classes, deep_supervision=self.deep_supervision) if self.num_dims == 2 else make_mednext3d(example_shape[0],
                                                                                                                                                    self.num_classes, 
                                                                                                                                                    deep_supervision=self.deep_supervision)
        
    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor,
                 toolbox: TrainerToolbox) -> tuple[float, dict[str, float]]:
        outputs = toolbox.model(images)

        if self.deep_supervision:
            total_loss = 0
            for output in outputs:
                target = F.interpolate(labels.float(), size=output.shape[2:], mode='nearest')
                loss, _ = toolbox.criterion(output, target)
                total_loss += loss
            total_loss /= len(outputs)

            final_output = outputs[-1]
            _, metrics = toolbox.criterion(final_output, labels)
        else:
            total_loss, metrics = toolbox.criterion(outputs, labels)

        total_loss.backward()
        return total_loss.item(), metrics
    
    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        outputs = (toolbox.ema if toolbox.ema else toolbox.model)(image)

        loss, metrics = toolbox.criterion(outputs, label)
        return -loss.item(), metrics, outputs.squeeze(0)
