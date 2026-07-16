"""Proteome preprocessing and domain-aware data loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .common import MissingStrategy, validate_proteome_frame


class ProteomeStandardizer:
    """Training-set imputation and per-protein standardization."""

    def __init__(
        self,
        *,
        impute: MissingStrategy = "median",
        with_mean: bool = True,
        with_std: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        if impute not in {"error", "median", "zero"}:
            raise ValueError(f"Unsupported imputation strategy: {impute!r}")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.impute = impute
        self.with_mean = with_mean
        self.with_std = with_std
        self.epsilon = float(epsilon)

        self.n_features_in_: int | None = None
        self.imputation_values_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray | pd.DataFrame) -> ProteomeStandardizer:
        array = self._as_2d_array(values)
        finite = np.isfinite(array)

        if self.impute == "error":
            if not finite.all():
                raise ValueError("Training matrix contains non-finite values")
            imputation_values = np.zeros(array.shape[1], dtype=np.float64)
            filled = array.copy()
        elif self.impute == "zero":
            imputation_values = np.zeros(array.shape[1], dtype=np.float64)
            filled = np.where(finite, array, 0.0)
        else:
            with np.errstate(all="ignore"):
                imputation_values = np.nanmedian(
                    np.where(finite, array, np.nan),
                    axis=0,
                )
            if not np.isfinite(imputation_values).all():
                bad_columns = np.flatnonzero(~np.isfinite(imputation_values)).tolist()
                raise ValueError(
                    "Cannot median-impute all-missing training proteins at columns "
                    f"{bad_columns[:10]}"
                )
            filled = self._fill_missing(array, imputation_values)

        mean = (
            filled.mean(axis=0)
            if self.with_mean
            else np.zeros(array.shape[1], dtype=np.float64)
        )
        if self.with_std:
            scale = filled.std(axis=0, ddof=0)
            scale = np.where(scale > self.epsilon, scale, 1.0)
        else:
            scale = np.ones(array.shape[1], dtype=np.float64)

        self.n_features_in_ = array.shape[1]
        self.imputation_values_ = imputation_values.astype(np.float64, copy=True)
        self.mean_ = mean.astype(np.float64, copy=True)
        self.scale_ = scale.astype(np.float64, copy=True)
        return self

    def transform(self, values: np.ndarray | pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        array = self._as_2d_array(values)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} proteins, received {array.shape[1]}"
            )

        finite = np.isfinite(array)
        if self.impute == "error" and not finite.all():
            raise ValueError("Matrix contains non-finite values")

        assert self.imputation_values_ is not None
        assert self.mean_ is not None
        assert self.scale_ is not None

        filled = self._fill_missing(array, self.imputation_values_)
        transformed = (filled - self.mean_) / self.scale_
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, values: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.fit(values).transform(values)

    def inverse_transform(self, values: np.ndarray | Tensor) -> np.ndarray:
        self._check_fitted()
        array = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(
            values
        )
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("values have an incompatible shape")

        assert self.mean_ is not None
        assert self.scale_ is not None
        return array.astype(np.float64) * self.scale_ + self.mean_

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        assert self.imputation_values_ is not None
        assert self.mean_ is not None
        assert self.scale_ is not None

        return {
            "impute": self.impute,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "epsilon": self.epsilon,
            "n_features_in": self.n_features_in_,
            "imputation_values": self.imputation_values_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ProteomeStandardizer:
        transformer = cls(
            impute=state["impute"],
            with_mean=bool(state["with_mean"]),
            with_std=bool(state["with_std"]),
            epsilon=float(state["epsilon"]),
        )
        transformer.n_features_in_ = int(state["n_features_in"])
        transformer.imputation_values_ = np.asarray(
            state["imputation_values"],
            dtype=np.float64,
        )
        transformer.mean_ = np.asarray(state["mean"], dtype=np.float64)
        transformer.scale_ = np.asarray(state["scale"], dtype=np.float64)
        return transformer

    @staticmethod
    def _as_2d_array(values: np.ndarray | pd.DataFrame) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("values must be two-dimensional")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("values cannot be empty")
        return array

    @staticmethod
    def _fill_missing(values: np.ndarray, replacements: np.ndarray) -> np.ndarray:
        filled = values.copy()
        rows, columns = np.where(~np.isfinite(filled))
        if rows.size:
            filled[rows, columns] = replacements[columns]
        return filled

    def _check_fitted(self) -> None:
        if any(
            item is None
            for item in (
                self.n_features_in_,
                self.imputation_values_,
                self.mean_,
                self.scale_,
            )
        ):
            raise RuntimeError("ProteomeStandardizer has not been fitted")


@dataclass
class DomainDataBundle:
    """Prepared loaders, preprocessing state, and sample metadata."""

    train_loader: DataLoader[tuple[Tensor, Tensor]]
    validation_loader: DataLoader[tuple[Tensor, Tensor]] | None
    transformer: ProteomeStandardizer
    feature_names: tuple[str, ...]
    domain_names: tuple[str, ...]
    sample_table: pd.DataFrame
    domain_class_weights: np.ndarray


def _stratified_train_validation_indices(
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)

        if validation_fraction == 0.0:
            n_validation = 0
        else:
            if indices.size < 2:
                raise ValueError(
                    f"Domain {label} needs at least 2 samples for validation splitting"
                )
            n_validation = max(1, int(round(indices.size * validation_fraction)))
            n_validation = min(n_validation, indices.size - 1)

        validation_parts.append(indices[:n_validation])
        train_parts.append(indices[n_validation:])

    train_indices = np.concatenate(train_parts)
    validation_indices = (
        np.concatenate(validation_parts)
        if any(part.size for part in validation_parts)
        else np.empty(0, dtype=np.int64)
    )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def _resolve_feature_names(
    reference: pd.DataFrame,
    neighborhood: pd.DataFrame,
    proteins: Sequence[str] | None,
) -> list[str]:
    if proteins is None:
        neighborhood_columns = set(neighborhood.columns)
        feature_names = [
            protein for protein in reference.columns if protein in neighborhood_columns
        ]
    else:
        feature_names = [str(protein) for protein in proteins]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("proteins contains duplicates")

        missing_reference = sorted(set(feature_names).difference(reference.columns))
        missing_neighborhood = sorted(
            set(feature_names).difference(neighborhood.columns)
        )
        if missing_reference or missing_neighborhood:
            raise KeyError(
                "Requested proteins are absent. "
                f"reference missing={missing_reference[:5]}, "
                f"neighborhood missing={missing_neighborhood[:5]}"
            )

    if not feature_names:
        raise ValueError("No shared proteins were supplied")
    return feature_names


def prepare_domain_data(
    reference: pd.DataFrame,
    neighborhood: pd.DataFrame,
    *,
    proteins: Sequence[str] | None = None,
    domain_names: tuple[str, str] = ("in_vitro_vsmc", "dvp_neighborhood"),
    validation_fraction: float = 0.2,
    batch_size: int = 64,
    seed: int = 13,
    impute: MissingStrategy = "median",
    standardize: bool = True,
    balance_training_domains: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DomainDataBundle:
    """Align, split, preprocess, and load two proteome domains."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(domain_names) != 2 or domain_names[0] == domain_names[1]:
        raise ValueError("domain_names must contain two distinct names")

    reference = validate_proteome_frame(reference, "reference")
    neighborhood = validate_proteome_frame(neighborhood, "neighborhood")
    feature_names = _resolve_feature_names(reference, neighborhood, proteins)

    reference_values = reference.loc[:, feature_names].to_numpy(dtype=np.float64)
    neighborhood_values = neighborhood.loc[:, feature_names].to_numpy(
        dtype=np.float64
    )
    values = np.vstack((reference_values, neighborhood_values))
    labels = np.concatenate(
        (
            np.zeros(reference.shape[0], dtype=np.int64),
            np.ones(neighborhood.shape[0], dtype=np.int64),
        )
    )

    sample_table = pd.DataFrame(
        {
            "sample_id": [str(index) for index in reference.index]
            + [str(index) for index in neighborhood.index],
            "domain": [domain_names[0]] * reference.shape[0]
            + [domain_names[1]] * neighborhood.shape[0],
            "domain_index": labels,
        }
    )

    train_indices, validation_indices = _stratified_train_validation_indices(
        labels,
        validation_fraction,
        seed,
    )
    sample_table["split"] = "train"
    if validation_indices.size:
        sample_table.loc[validation_indices, "split"] = "validation"

    transformer = ProteomeStandardizer(
        impute=impute,
        with_mean=standardize,
        with_std=standardize,
    )
    transformer.fit(values[train_indices])
    transformed = transformer.transform(values)

    x_tensor = torch.from_numpy(transformed)
    domain_tensor = torch.from_numpy(labels)
    train_dataset = TensorDataset(
        x_tensor[train_indices],
        domain_tensor[train_indices],
    )

    generator = torch.Generator().manual_seed(seed)
    sampler: WeightedRandomSampler | None
    if balance_training_domains:
        train_labels = labels[train_indices]
        counts = np.bincount(train_labels, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError("Each domain must be represented in the training split")

        sample_weights = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator if sampler is None else None,
    )

    validation_loader: DataLoader[tuple[Tensor, Tensor]] | None = None
    if validation_indices.size:
        validation_dataset = TensorDataset(
            x_tensor[validation_indices],
            domain_tensor[validation_indices],
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    training_counts = np.bincount(labels[train_indices], minlength=2).astype(
        np.float64
    )
    if balance_training_domains:
        class_weights = np.ones(2, dtype=np.float64)
    else:
        class_weights = training_counts.sum() / (2.0 * training_counts)

    return DomainDataBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        transformer=transformer,
        feature_names=tuple(feature_names),
        domain_names=tuple(domain_names),
        sample_table=sample_table,
        domain_class_weights=class_weights.astype(np.float32),
    )
