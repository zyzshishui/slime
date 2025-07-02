from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


@dataclass
class Sample:
    """The sample generated"""

    index: Optional[int] = None
    prompt: Optional[str] = None
    label: Optional[str] = None
    response: Optional[str] = None
    tokens: Optional[list[int]] = None
    response_length: Optional[int] = None
    completion_tokens: Optional[int] = None
    reward: Optional[float] = None
    loss_mask: Optional[list[int]] = None
    metadata: dict = field(default_factory=dict)
    version: int = 0

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.PENDING


@dataclass
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int
