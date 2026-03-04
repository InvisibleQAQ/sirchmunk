"""Visual Grep operator package — pluggable scoring operators.

Each operator is a :class:`GrepOperator` subclass that scores a single
:class:`ImageSignature` against a :class:`VisualConstraint` and returns
a relevance score in [0, 1].
"""

from .base import GrepOperator
from .block_hash_op import BlockHashOperator
from .color_op import ColorOperator
from .contrast_op import ContrastiveColorOperator
from .entropy_op import EntropyOperator
from .gist_op import GISTOperator
from .metadata_op import MetadataOperator
from .negative_op import NegativeOperator
from .siglip_embed_op import SigLIPEmbedOperator

DEFAULT_OPERATORS: list[GrepOperator] = [
    ColorOperator(),
    MetadataOperator(),
    BlockHashOperator(),
    ContrastiveColorOperator(),
    GISTOperator(),
    EntropyOperator(),
    NegativeOperator(),
    SigLIPEmbedOperator(),
]

__all__ = [
    "GrepOperator",
    "ColorOperator",
    "MetadataOperator",
    "BlockHashOperator",
    "ContrastiveColorOperator",
    "GISTOperator",
    "EntropyOperator",
    "NegativeOperator",
    "SigLIPEmbedOperator",
    "DEFAULT_OPERATORS",
]
