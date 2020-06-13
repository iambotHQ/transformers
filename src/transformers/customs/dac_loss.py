from typing import Any, Optional

import torch
from logzero import logger
from overrides import overrides
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss, _Loss


class DACLoss(_Loss):
    name: str = "dac"
    """
    Implementation of DAC loss function with (NOT YET) implemented parameter auto tuner
    Article: https://openreview.net/pdf?id=rJxF73R9tX
    """

    def __init__(
        self, alpha: float, auto_tune: bool = False, reduction: str = "mean", weight: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.reductor = getattr(torch, reduction, None)
        self.weight = weight

        if auto_tune:
            logger.warning("Auto tune is not yet implemented. Using casual mode.")

    @overrides
    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inputs = inputs.softmax(-1)

        one_hot: torch.Tensor = torch.zeros(inputs.shape, device=inputs.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(1), 1.0)[:, :-1].to(inputs.device)

        casual_class_logits: torch.Tensor = inputs[:, :-1]
        abstention_class_logits: torch.Tensor = torch.tensor(1.0, device=inputs.device) - inputs[:, -1]

        loss: torch.Tensor = torch.div(casual_class_logits, abstention_class_logits.unsqueeze(dim=1)).log()
        loss = torch.mul(loss, one_hot).sum(dim=1)
        loss = torch.neg(loss)
        loss = torch.mul(loss, abstention_class_logits)
        penelty: torch.Tensor = torch.div(1.0, abstention_class_logits).log()
        penelty = torch.mul(self.alpha, penelty)
        loss = torch.add(loss, penelty)

        return loss if self.reductor is None else self.reductor(loss)

    def _auto_tune(self) -> None:
        raise NotImplementedError
