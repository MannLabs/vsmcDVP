"""Latent embedding, reconstruction, and cross-domain decoding."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .common import resolve_device
from .model import DomainAdversarialCVAE
from .preprocessing import ProteomeStandardizer


def _align_inference_matrix(
    matrix: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, pd.Index]:
    if not isinstance(matrix, pd.DataFrame):
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if values.shape[0] == 0:
            raise ValueError("matrix must contain at least one profile")
        return values, pd.RangeIndex(values.shape[0], name="sample")

    if not matrix.index.is_unique:
        raise ValueError("Inference matrix contains duplicate sample IDs")

    working = matrix.copy()
    working.columns = working.columns.astype(str)
    if not working.columns.is_unique:
        duplicates = working.columns[working.columns.duplicated()].tolist()
        raise ValueError(
            "Inference protein IDs are not unique after string conversion: "
            f"{duplicates[:5]}"
        )

    if feature_names is not None:
        missing = sorted(set(feature_names).difference(working.columns))
        if missing:
            raise KeyError(f"Inference matrix is missing proteins: {missing[:10]}")
        working = working.loc[:, list(feature_names)]

    values = working.to_numpy(dtype=np.float64)
    if values.shape[0] == 0:
        raise ValueError("matrix must contain at least one profile")
    return values, matrix.index.copy()


def _domain_array(
    domains: int | Sequence[int] | np.ndarray,
    n_rows: int,
) -> np.ndarray:
    if np.isscalar(domains):
        return np.full(n_rows, int(domains), dtype=np.int64)

    labels = np.asarray(domains, dtype=np.int64)
    if labels.ndim != 1 or labels.size != n_rows:
        raise ValueError("domains must be a scalar or one label per row")
    return labels


def _prepare_inference_values(
    matrix: pd.DataFrame | np.ndarray,
    *,
    transformer: ProteomeStandardizer | None,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, pd.Index]:
    values, index = _align_inference_matrix(matrix, feature_names)

    if transformer is not None:
        values = transformer.transform(values)
    else:
        if not np.isfinite(values).all():
            raise ValueError("Inference values contain non-finite entries")
        values = values.astype(np.float32, copy=False)

    return values, index


def _validate_domains(
    labels: np.ndarray,
    model: DomainAdversarialCVAE,
    name: str,
) -> None:
    if labels.size and (labels.min() < 0 or labels.max() >= model.n_domains):
        raise ValueError(f"A {name} domain label is outside the configured range")


@torch.inference_mode()
def encode_proteomes(
    model: DomainAdversarialCVAE,
    matrix: pd.DataFrame | np.ndarray,
    domains: int | Sequence[int] | np.ndarray,
    *,
    transformer: ProteomeStandardizer | None = None,
    feature_names: Sequence[str] | None = None,
    batch_size: int = 256,
    device: str | None = None,
    include_posterior_sd: bool = False,
) -> pd.DataFrame:
    """Return posterior-mean latent embeddings for proteome profiles."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    values, index = _prepare_inference_values(
        matrix,
        transformer=transformer,
        feature_names=feature_names,
    )
    if values.shape[1] != model.input_dim:
        raise ValueError(
            f"Model expects {model.input_dim} proteins, received {values.shape[1]}"
        )

    labels = _domain_array(domains, values.shape[0])
    _validate_domains(labels, model, "source")

    target_device = (
        resolve_device(device) if device is not None else next(model.parameters()).device
    )
    model = model.to(target_device)
    model.eval()

    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []

    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        x = torch.as_tensor(
            values[start:stop],
            dtype=torch.float32,
            device=target_device,
        )
        domain = torch.as_tensor(
            labels[start:stop],
            dtype=torch.long,
            device=target_device,
        )
        mu, logvar = model.encode(x, domain)
        means.append(mu.cpu().numpy())
        if include_posterior_sd:
            standard_deviations.append(torch.exp(0.5 * logvar).cpu().numpy())

    result = pd.DataFrame(
        np.vstack(means),
        index=index,
        columns=[f"z_{i + 1}" for i in range(model.latent_dim)],
    )
    if include_posterior_sd:
        posterior_sd = np.vstack(standard_deviations)
        for i in range(model.latent_dim):
            result[f"z_sd_{i + 1}"] = posterior_sd[:, i]

    result.insert(0, "domain_index", labels)
    return result


@torch.inference_mode()
def reconstruct_or_translate_proteomes(
    model: DomainAdversarialCVAE,
    matrix: pd.DataFrame | np.ndarray,
    source_domains: int | Sequence[int] | np.ndarray,
    *,
    target_domains: int | Sequence[int] | np.ndarray | None = None,
    transformer: ProteomeStandardizer | None = None,
    feature_names: Sequence[str] | None = None,
    batch_size: int = 256,
    device: str | None = None,
) -> pd.DataFrame:
    """Reconstruct profiles or decode them under another domain condition."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    transformed, index = _prepare_inference_values(
        matrix,
        transformer=transformer,
        feature_names=feature_names,
    )
    if transformed.shape[1] != model.input_dim:
        raise ValueError(
            f"Model expects {model.input_dim} proteins, received {transformed.shape[1]}"
        )

    source = _domain_array(source_domains, transformed.shape[0])
    target = (
        source.copy()
        if target_domains is None
        else _domain_array(target_domains, transformed.shape[0])
    )
    _validate_domains(source, model, "source")
    _validate_domains(target, model, "target")

    target_device = (
        resolve_device(device) if device is not None else next(model.parameters()).device
    )
    model = model.to(target_device)
    model.eval()

    decoded_batches: list[np.ndarray] = []
    for start in range(0, transformed.shape[0], batch_size):
        stop = min(start + batch_size, transformed.shape[0])
        x = torch.as_tensor(
            transformed[start:stop],
            dtype=torch.float32,
            device=target_device,
        )
        source_domain = torch.as_tensor(
            source[start:stop],
            dtype=torch.long,
            device=target_device,
        )
        target_domain = torch.as_tensor(
            target[start:stop],
            dtype=torch.long,
            device=target_device,
        )

        mu, _ = model.encode(x, source_domain)
        decoded_batches.append(model.decode(mu, target_domain).cpu().numpy())

    reconstructed = np.vstack(decoded_batches)
    if transformer is not None:
        reconstructed = transformer.inverse_transform(reconstructed)

    if feature_names is not None:
        columns = list(feature_names)
    elif isinstance(matrix, pd.DataFrame):
        columns = [str(column) for column in matrix.columns]
    else:
        columns = [f"protein_{i + 1}" for i in range(model.input_dim)]

    return pd.DataFrame(reconstructed, index=index, columns=columns)
