from typing import Optional
import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss


class DiceLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-6,
        size_average: Optional[torch.Tensor] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super(DiceLoss, self).__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
        self.register_buffer("eps", torch.tensor(epsilon))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.softmax(output, dim=1)

        if self.weight is None:
            intersection = torch.mean(target * output)
            sum_ = torch.mean(target) + torch.mean(output)
        else:
            mean_dims = tuple(
                [0] + list(range(2, target.dim()))
            )  # -> (0,2,3) in segm or (0) in classif
            intersection = (
                1.0
                / len(self.weight)
                * torch.dot(torch.mean(target * output, mean_dims), self.weight)
            )
            sum_ = (
                1.0
                / len(self.weight)
                * torch.dot(torch.mean(target + output, mean_dims), self.weight)
            )
        loss = 1.0 - (2.0 * intersection + self.eps) / (sum_ + self.eps)
        return loss


class TrainingLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[list] = None,
    ):
        super(TrainingLoss, self).__init__()
        
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
        self.dice = DiceLoss(weight=self.class_weights)
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.cross_entropy(output, target) + self.dice(output, target)
