"""Checkpoint serialization for models and preprocessing state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from .common import PathLike
from .model import CVAEConfig, DomainAdversarialCVAE
from .preprocessing import ProteomeStandardizer
from .training import TrainingConfig

BUNDLE_FORMAT_VERSION = 1


@dataclass
class LoadedCVAEBundle:
    model: DomainAdversarialCVAE
    feature_names: tuple[str, ...]
    domain_names: tuple[str, ...]
    transformer: ProteomeStandardizer | None
    training_config: TrainingConfig | None
    history: pd.DataFrame | None
    extra: dict[str, Any]


def save_cvae_bundle(
    path: PathLike,
    model: DomainAdversarialCVAE,
    *,
    feature_names: Sequence[str],
    domain_names: Sequence[str],
    transformer: ProteomeStandardizer | None = None,
    training_config: TrainingConfig | None = None,
    history: pd.DataFrame | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save model weights and the metadata required for aligned inference."""

    if len(feature_names) != model.input_dim:
        raise ValueError("feature_names length does not match model.input_dim")
    if len(domain_names) != model.n_domains:
        raise ValueError("domain_names length does not match model.n_domains")

    payload: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model_config": asdict(model.config),
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "feature_names": [str(name) for name in feature_names],
        "domain_names": [str(name) for name in domain_names],
        "transformer": transformer.state_dict() if transformer is not None else None,
        "training_config": (
            asdict(training_config) if training_config is not None else None
        ),
        "history": history.to_dict(orient="records") if history is not None else None,
        "extra_json": json.dumps(dict(extra or {}), default=str),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_cvae_bundle(
    path: PathLike,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCVAEBundle:
    """Load a bundle written by :func:`save_cvae_bundle`."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        payload = torch.load(path, map_location=map_location)

    if payload.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported CVAE bundle format")

    model_config_data = dict(payload["model_config"])
    for key in ("encoder_hidden", "decoder_hidden", "discriminator_hidden"):
        model_config_data[key] = tuple(model_config_data[key])

    model = DomainAdversarialCVAE(CVAEConfig(**model_config_data))
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    model.eval()

    transformer_state = payload.get("transformer")
    transformer = (
        ProteomeStandardizer.from_state_dict(transformer_state)
        if transformer_state is not None
        else None
    )

    training_config_state = payload.get("training_config")
    training_config = (
        TrainingConfig(**training_config_state)
        if training_config_state is not None
        else None
    )

    history_records = payload.get("history")
    history = (
        pd.DataFrame.from_records(history_records)
        if history_records is not None
        else None
    )

    return LoadedCVAEBundle(
        model=model,
        feature_names=tuple(payload["feature_names"]),
        domain_names=tuple(payload["domain_names"]),
        transformer=transformer,
        training_config=training_config,
        history=history,
        extra=json.loads(payload.get("extra_json", "{}")),
    )
