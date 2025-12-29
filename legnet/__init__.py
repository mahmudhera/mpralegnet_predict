"""legnet: minimal MPRA-LegNet (LegNet) loader + predictor + fine-tuning utilities."""

from .model import LegNet
from .config import LegNetConfig
from .encoding import encode_seq, encode_many, reverse_complement
from .checkpoint import load_model, save_checkpoint, resolve_checkpoint_path, resolve_config_path
from .data import (
    SequenceRegressionDataset,
    IndexedSequenceRegressionDataset,
    SequenceOnlyDataset,
    split_indices,
)
from .metrics import pearsonr

__all__ = [
    "LegNet",
    "LegNetConfig",
    "encode_seq",
    "encode_many",
    "reverse_complement",
    "load_model",
    "save_checkpoint",
    "resolve_checkpoint_path",
    "resolve_config_path",
    "SequenceRegressionDataset",
    "IndexedSequenceRegressionDataset",
    "SequenceOnlyDataset",
    "split_indices",
    "pearsonr",
]
