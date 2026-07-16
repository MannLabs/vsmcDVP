"""Utilities for reshaping DIA-NN reports."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .common import PathLike


def diann_report_to_protein_matrix(
    report: PathLike | pd.DataFrame,
    *,
    sample_col: str = "Run",
    protein_col: str = "Protein.Group",
    quantity_col: str = "PG.MaxLFQ",
    q_value_col: str | None = "PG.Q.Value",
    max_q_value: float | None = 0.01,
    aggregation: Literal["median", "mean", "sum", "max"] = "median",
    log2_transform: bool = True,
) -> pd.DataFrame:
    """Pivot a long DIA-NN report into a sample-by-protein matrix.

    Duplicate sample/protein rows are aggregated before the optional log2
    transform. Non-positive quantities are treated as missing on log2 scale.
    """

    if isinstance(report, pd.DataFrame):
        frame = report.copy()
    else:
        frame = pd.read_csv(report, sep="\t", low_memory=False)

    required = {sample_col, protein_col, quantity_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"DIA-NN report is missing required columns: {sorted(missing)}")

    if q_value_col is not None and max_q_value is not None:
        if q_value_col not in frame.columns:
            raise KeyError(
                f"q-value column {q_value_col!r} was requested but is absent"
            )
        q_values = pd.to_numeric(frame[q_value_col], errors="coerce")
        frame = frame.loc[q_values <= float(max_q_value)].copy()

    frame[quantity_col] = pd.to_numeric(frame[quantity_col], errors="coerce")
    frame = frame.dropna(subset=[sample_col, protein_col, quantity_col])

    grouped = (
        frame.groupby([sample_col, protein_col], observed=True, sort=False)[quantity_col]
        .agg(aggregation)
        .unstack(protein_col)
    )
    grouped.index = grouped.index.astype(str)
    grouped.columns = grouped.columns.astype(str)

    if log2_transform:
        grouped = np.log2(grouped.where(grouped > 0.0))

    grouped.index.name = "sample"
    grouped.columns.name = "protein"
    return grouped
