"""Cross-domain protein co-expression network selection."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .common import MissingStrategy, PathLike, validate_proteome_frame


@dataclass(frozen=True)
class NetworkFeatureSelectionResult:
    """Result of cross-domain co-expression-network feature selection."""

    shared_proteins: tuple[str, ...]
    network_eligible_proteins: tuple[str, ...]
    selected_proteins: tuple[str, ...]
    metrics: pd.DataFrame
    threshold: float

    @property
    def n_shared(self) -> int:
        return len(self.shared_proteins)

    @property
    def n_network_eligible(self) -> int:
        return len(self.network_eligible_proteins)

    @property
    def n_selected(self) -> int:
        return len(self.selected_proteins)

    def save_metrics(self, path: PathLike) -> None:
        """Write the per-protein feature-selection table as TSV."""

        self.metrics.to_csv(path, sep="\t", index=True)


def _impute_network_matrix(
    values: np.ndarray,
    strategy: MissingStrategy,
    *,
    matrix_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute a network matrix and return all-missing-column flags."""

    values = np.asarray(values, dtype=np.float64)
    all_missing = ~np.isfinite(values).any(axis=0)

    if strategy == "error":
        if not np.isfinite(values).all():
            count = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"{matrix_name} contains {count} non-finite values; choose "
                "missing='median' or missing='zero', or impute upstream"
            )
        return values, all_missing

    result = values.copy()
    if strategy == "median":
        with np.errstate(all="ignore"):
            medians = np.nanmedian(np.where(np.isfinite(result), result, np.nan), axis=0)
        rows, cols = np.where(~np.isfinite(result))
        if rows.size:
            result[rows, cols] = medians[cols]
    elif strategy == "zero":
        result[~np.isfinite(result)] = 0.0
    else:
        raise ValueError(f"Unsupported missing strategy: {strategy!r}")

    return result, all_missing


def _spearman_correlation_matrix(
    values: np.ndarray,
    *,
    block_size: int = 512,
    dtype: np.dtype[Any] = np.dtype(np.float32),
    output_path: Path | None = None,
) -> np.ndarray:
    """Compute a protein-by-protein Spearman matrix in column blocks."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("Spearman matrix calculation requires finite values")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_profiles, n_proteins = values.shape
    if n_profiles < 3 or n_proteins < 2:
        raise ValueError("At least 3 profiles and 2 proteins are required")

    ranks = rankdata(values, axis=0, method="average").astype(dtype, copy=False)
    ranks -= ranks.mean(axis=0, keepdims=True, dtype=np.float64).astype(dtype)
    norms = np.sqrt(np.sum(ranks * ranks, axis=0, dtype=np.float64))
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Constant protein columns reached Spearman calculation")
    ranks /= norms.astype(dtype, copy=False)

    if output_path is None:
        correlations: np.ndarray = np.empty((n_proteins, n_proteins), dtype=dtype)
    else:
        correlations = np.memmap(
            output_path,
            mode="w+",
            dtype=dtype,
            shape=(n_proteins, n_proteins),
        )

    for start in range(0, n_proteins, block_size):
        stop = min(start + block_size, n_proteins)
        block = ranks[:, start:stop].T @ ranks
        np.clip(block, -1.0, 1.0, out=block)
        correlations[start:stop, :] = block

    if isinstance(correlations, np.memmap):
        correlations.flush()
    return correlations


def _rowwise_network_profile_similarity(
    correlation_a: np.ndarray,
    correlation_b: np.ndarray,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Spearman-correlate corresponding network rows, excluding self-edges."""

    if correlation_a.shape != correlation_b.shape:
        raise ValueError("The two correlation matrices must have identical shape")
    if correlation_a.ndim != 2 or correlation_a.shape[0] != correlation_a.shape[1]:
        raise ValueError("Correlation matrices must be square")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_proteins = correlation_a.shape[0]
    similarities = np.full(n_proteins, np.nan, dtype=np.float64)

    for start in range(0, n_proteins, block_size):
        stop = min(start + block_size, n_proteins)
        local_size = stop - start

        a = np.asarray(correlation_a[start:stop, :], dtype=np.float64).copy()
        b = np.asarray(correlation_b[start:stop, :], dtype=np.float64).copy()
        local_rows = np.arange(local_size)
        global_columns = np.arange(start, stop)
        a[local_rows, global_columns] = np.nan
        b[local_rows, global_columns] = np.nan

        # rankdata with nan_policy='omit' leaves the self-edge as NaN while
        # ranking all other co-expression edges independently in each domain.
        ranked_a = rankdata(
            a, axis=1, method="average", nan_policy="omit"
        ).astype(np.float64, copy=False)
        ranked_b = rankdata(
            b, axis=1, method="average", nan_policy="omit"
        ).astype(np.float64, copy=False)

        valid = np.isfinite(ranked_a) & np.isfinite(ranked_b)
        counts = valid.sum(axis=1)

        sum_a = np.where(valid, ranked_a, 0.0).sum(axis=1)
        sum_b = np.where(valid, ranked_b, 0.0).sum(axis=1)
        mean_a = np.divide(
            sum_a,
            counts,
            out=np.full(local_size, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        mean_b = np.divide(
            sum_b,
            counts,
            out=np.full(local_size, np.nan, dtype=np.float64),
            where=counts > 0,
        )

        centered_a = np.where(valid, ranked_a - mean_a[:, None], 0.0)
        centered_b = np.where(valid, ranked_b - mean_b[:, None], 0.0)
        numerator = np.einsum("ij,ij->i", centered_a, centered_b)
        denominator = np.sqrt(
            np.einsum("ij,ij->i", centered_a, centered_a)
            * np.einsum("ij,ij->i", centered_b, centered_b)
        )

        block_similarity = np.divide(
            numerator,
            denominator,
            out=np.full(local_size, np.nan, dtype=np.float64),
            where=(counts >= 3) & (denominator > 0.0),
        )
        similarities[start:stop] = block_similarity

    return similarities


def select_conserved_network_proteins(
    reference: pd.DataFrame,
    neighborhood: pd.DataFrame,
    *,
    threshold: float = 0.3,
    missing: MissingStrategy = "median",
    min_variance: float = 1e-12,
    correlation_block_size: int = 512,
    similarity_block_size: int = 128,
    correlation_dtype: Literal["float32", "float64"] = "float32",
    cache_dir: PathLike | None = None,
) -> NetworkFeatureSelectionResult:
    """Retain proteins whose co-expression edge profiles agree across domains.

    The self-edge is removed before comparing the two edge profiles. Retention
    uses the strict rule ``network_similarity > threshold``.
    """

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie between -1 and 1")
    if min_variance < 0.0:
        raise ValueError("min_variance must be non-negative")

    reference = validate_proteome_frame(reference, "reference")
    neighborhood = validate_proteome_frame(neighborhood, "neighborhood")

    neighborhood_columns = set(neighborhood.columns)
    shared = [protein for protein in reference.columns if protein in neighborhood_columns]
    if len(shared) < 4:
        raise ValueError(
            "At least 4 shared proteins are required to compare network profiles"
        )

    reference_shared = reference.loc[:, shared]
    neighborhood_shared = neighborhood.loc[:, shared]
    reference_values_raw = reference_shared.to_numpy(dtype=np.float64, copy=True)
    neighborhood_values_raw = neighborhood_shared.to_numpy(dtype=np.float64, copy=True)

    ref_nonmissing = np.isfinite(reference_values_raw).sum(axis=0)
    nbr_nonmissing = np.isfinite(neighborhood_values_raw).sum(axis=0)

    reference_values, ref_all_missing = _impute_network_matrix(
        reference_values_raw, missing, matrix_name="reference"
    )
    neighborhood_values, nbr_all_missing = _impute_network_matrix(
        neighborhood_values_raw, missing, matrix_name="neighborhood"
    )

    with np.errstate(invalid="ignore"):
        ref_variance = np.var(reference_values, axis=0)
        nbr_variance = np.var(neighborhood_values, axis=0)

    ref_finite = np.isfinite(reference_values).all(axis=0)
    nbr_finite = np.isfinite(neighborhood_values).all(axis=0)
    eligible = (
        ~ref_all_missing
        & ~nbr_all_missing
        & ref_finite
        & nbr_finite
        & (ref_variance > min_variance)
        & (nbr_variance > min_variance)
    )

    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size < 4:
        raise ValueError(
            "Fewer than 4 shared proteins remain after all-missing/variance filtering"
        )

    eligible_proteins = [shared[index] for index in eligible_indices]
    reference_network_values = reference_values[:, eligible_indices]
    neighborhood_network_values = neighborhood_values[:, eligible_indices]

    dtype = np.dtype(correlation_dtype)
    temp_context: tempfile.TemporaryDirectory[str] | None = None
    try:
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            temp_context = tempfile.TemporaryDirectory(
                prefix="vsmc_network_", dir=cache_path
            )
            workdir = Path(temp_context.name)
            corr_ref_path = workdir / "reference_corr.dat"
            corr_nbr_path = workdir / "neighborhood_corr.dat"
        else:
            corr_ref_path = None
            corr_nbr_path = None

        corr_reference = _spearman_correlation_matrix(
            reference_network_values,
            block_size=correlation_block_size,
            dtype=dtype,
            output_path=corr_ref_path,
        )
        corr_neighborhood = _spearman_correlation_matrix(
            neighborhood_network_values,
            block_size=correlation_block_size,
            dtype=dtype,
            output_path=corr_nbr_path,
        )
        similarities = _rowwise_network_profile_similarity(
            corr_reference,
            corr_neighborhood,
            block_size=similarity_block_size,
        )
    finally:
        # Explicitly release memmaps before the temporary directory is removed.
        if "corr_reference" in locals():
            del corr_reference
        if "corr_neighborhood" in locals():
            del corr_neighborhood
        if temp_context is not None:
            temp_context.cleanup()

    metrics = pd.DataFrame(
        {
            "reference_nonmissing": ref_nonmissing,
            "neighborhood_nonmissing": nbr_nonmissing,
            "reference_variance_after_imputation": ref_variance,
            "neighborhood_variance_after_imputation": nbr_variance,
            "network_eligible": eligible,
            "network_similarity": np.nan,
            "retained": False,
            "exclusion_reason": "",
        },
        index=pd.Index(shared, name="protein"),
    )
    metrics.loc[eligible_proteins, "network_similarity"] = similarities
    retained = np.isfinite(similarities) & (similarities > threshold)
    selected = [
        protein for protein, keep in zip(eligible_proteins, retained, strict=True) if keep
    ]
    metrics.loc[selected, "retained"] = True

    exclusion_reasons = np.full(len(shared), "", dtype=object)
    exclusion_reasons[ref_all_missing] = "all_missing_in_reference"
    exclusion_reasons[nbr_all_missing] = "all_missing_in_neighborhood"
    both_missing = ref_all_missing & nbr_all_missing
    exclusion_reasons[both_missing] = "all_missing_in_both_domains"

    nonfinite_after = ~(ref_finite & nbr_finite)
    exclusion_reasons[(exclusion_reasons == "") & nonfinite_after] = (
        "nonfinite_after_imputation"
    )
    low_ref = ref_variance <= min_variance
    low_nbr = nbr_variance <= min_variance
    exclusion_reasons[(exclusion_reasons == "") & low_ref & low_nbr] = (
        "constant_in_both_domains"
    )
    exclusion_reasons[(exclusion_reasons == "") & low_ref] = "constant_in_reference"
    exclusion_reasons[(exclusion_reasons == "") & low_nbr] = (
        "constant_in_neighborhood"
    )

    similarity_missing = eligible.copy()
    similarity_missing[eligible_indices] = ~np.isfinite(similarities)
    exclusion_reasons[(exclusion_reasons == "") & similarity_missing] = (
        "undefined_network_similarity"
    )
    not_retained = eligible.copy()
    not_retained[eligible_indices] = np.isfinite(similarities) & ~retained
    exclusion_reasons[(exclusion_reasons == "") & not_retained] = (
        "network_similarity_at_or_below_threshold"
    )
    metrics["exclusion_reason"] = exclusion_reasons

    metrics = metrics.sort_values(
        ["retained", "network_similarity"], ascending=[False, False], na_position="last"
    )

    return NetworkFeatureSelectionResult(
        shared_proteins=tuple(shared),
        network_eligible_proteins=tuple(eligible_proteins),
        selected_proteins=tuple(selected),
        metrics=metrics,
        threshold=float(threshold),
    )
