"""Loading feature Parquet into numpy/torch tensors for model training."""

from __future__ import annotations

import logging

import numpy as np
import torch
from pyspark.sql import functions as F
from torch.utils.data import TensorDataset

from ctr.data.synthetic import CATEGORICAL_FEATURES, LABEL_COLUMN
from ctr.features.transforms import NUMERIC_BLOCK_COLUMN

logger = logging.getLogger(__name__)


def index_columns(fields: list[str] | None = None) -> list[str]:
    """Feature-column names for a set of categorical fields."""
    return [f"{field}_idx" for field in (fields or CATEGORICAL_FEATURES)]


def load_split_arrays(
    spark,
    path: str,
    categorical_fields: list[str] | None = None,
    positives_only: bool = False,
) -> dict[str, np.ndarray]:
    """Read one features split into numpy arrays.

    Returns ``cats`` [N, F] int64 (in ``categorical_fields`` order),
    ``numeric`` [N, D] float32, ``labels`` [N] float32 and the field list.
    ``positives_only`` keeps only clicked rows (used for two-tower training
    where negatives come from the batch itself).
    """
    fields = list(categorical_fields or CATEGORICAL_FEATURES)
    columns = [LABEL_COLUMN, NUMERIC_BLOCK_COLUMN] + index_columns(fields)
    df = spark.read.parquet(path).select(*columns)
    if positives_only:
        df = df.filter(F.col(LABEL_COLUMN) == 1)
    pdf = df.toPandas()

    cats = pdf[index_columns(fields)].to_numpy(dtype=np.int64)
    numeric = np.stack(pdf[NUMERIC_BLOCK_COLUMN].to_numpy(), axis=0).astype(
        np.float32
    )
    labels = pdf[LABEL_COLUMN].to_numpy(dtype=np.float32)
    return {"cats": cats, "numeric": numeric, "labels": labels, "fields": fields}


def to_tensor_dataset(split: dict[str, np.ndarray]) -> TensorDataset:
    """Wrap a split's arrays into a (cats, numeric, labels) TensorDataset."""
    return TensorDataset(
        torch.from_numpy(split["cats"]),
        torch.from_numpy(split["numeric"]),
        torch.from_numpy(split["labels"]),
    )
