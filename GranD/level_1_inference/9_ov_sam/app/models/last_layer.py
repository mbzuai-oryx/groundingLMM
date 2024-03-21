from typing import Tuple

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class LastLayerNeck(BaseModule):
    r"""Last Layer Neck

    Return the last layer feature of the backbone.
    """

    def __init__(self) -> None:
        super().__init__(init_cfg=None)

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        return inputs[-1]
