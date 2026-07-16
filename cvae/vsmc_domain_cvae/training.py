"""Alternating minimax training for the domain-adversarial CVAE."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .common import resolve_device, set_global_seed
from .model import CVAEOutput, DomainAdversarialCVAE

logger = logging.getLogger(__name__)

MonitorMetric = Literal[
    "validation_reconstruction",
    "validation_autoencoder_objective",
    "train_reconstruction",
    "train_autoencoder_objective",
]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    autoencoder_learning_rate: float = 1e-3
    discriminator_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    reconstruction_weight: float = 1.0
    beta_kl: float = 0.01
    lambda_adversarial: float = 0.1
    lambda_l1: float = 1e-4
    kl_warmup_epochs: int = 20
    adversarial_warmup_epochs: int = 20
    discriminator_steps: int = 1
    gradient_clip_norm: float | None = 5.0
    early_stopping_patience: int | None = 30
    early_stopping_min_delta: float = 1e-4
    monitor: MonitorMetric = "validation_reconstruction"
    restore_best_weights: bool = True
    seed: int = 13
    deterministic: bool = False
    device: str | None = None
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.autoencoder_learning_rate <= 0:
            raise ValueError("autoencoder_learning_rate must be positive")
        if self.discriminator_learning_rate <= 0:
            raise ValueError("discriminator_learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        for name in (
            "reconstruction_weight",
            "beta_kl",
            "lambda_adversarial",
            "lambda_l1",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.kl_warmup_epochs < 0 or self.adversarial_warmup_epochs < 0:
            raise ValueError("warmup epochs must be non-negative")
        if self.discriminator_steps <= 0:
            raise ValueError("discriminator_steps must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError("early_stopping_patience must be positive or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        valid_monitors = {
            "validation_reconstruction",
            "validation_autoencoder_objective",
            "train_reconstruction",
            "train_autoencoder_objective",
        }
        if self.monitor not in valid_monitors:
            raise ValueError(f"Unsupported monitor: {self.monitor!r}")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


@dataclass(frozen=True)
class LossTerms:
    reconstruction: Tensor
    kl: Tensor
    sparsity: Tensor
    domain: Tensor
    autoencoder_objective: Tensor


def _kl_divergence_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def _calculate_loss_terms(
    output: CVAEOutput,
    target: Tensor,
    domain: Tensor,
    *,
    reconstruction_weight: float,
    beta_kl: float,
    lambda_l1: float,
    lambda_adversarial: float,
    domain_class_weights: Tensor | None,
) -> LossTerms:
    reconstruction = F.mse_loss(output.reconstruction, target, reduction="mean")
    kl = _kl_divergence_standard_normal(output.mu, output.logvar)
    sparsity = output.z.abs().mean()
    domain_loss = F.cross_entropy(
        output.domain_logits,
        domain,
        weight=domain_class_weights,
    )
    objective = (
        reconstruction_weight * reconstruction
        + beta_kl * kl
        + lambda_l1 * sparsity
        - lambda_adversarial * domain_loss
    )
    return LossTerms(reconstruction, kl, sparsity, domain_loss, objective)


def _ramped_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target)
    progress = min(max(epoch, 0) / warmup_epochs, 1.0)
    return float(target * progress)


class _MetricAccumulator:
    def __init__(self) -> None:
        self._weighted_sums: dict[str, float] = {}
        self._sample_count = 0

    def add(self, batch_size: int, **metrics: float) -> None:
        self._sample_count += batch_size
        for name, value in metrics.items():
            self._weighted_sums[name] = (
                self._weighted_sums.get(name, 0.0) + float(value) * batch_size
            )

    def means(self) -> dict[str, float]:
        if self._sample_count == 0:
            raise RuntimeError("No batches were accumulated")
        return {
            name: total / self._sample_count
            for name, total in self._weighted_sums.items()
        }


class CVAETrainer:
    """Alternating minimax trainer for ``DomainAdversarialCVAE``."""

    def __init__(
        self,
        model: DomainAdversarialCVAE,
        config: TrainingConfig,
        *,
        domain_class_weights: Sequence[float] | np.ndarray | Tensor | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)

        self.domain_class_weights = self._prepare_class_weights(domain_class_weights)
        self.autoencoder_optimizer = torch.optim.AdamW(
            list(model.autoencoder_parameters()),
            lr=config.autoencoder_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            list(model.discriminator_parameters()),
            lr=config.discriminator_learning_rate,
            weight_decay=config.weight_decay,
        )

        self.history_: pd.DataFrame | None = None
        self.best_epoch_: int | None = None
        self.best_monitor_value_: float | None = None

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        validation_loader: DataLoader[tuple[Tensor, Tensor]] | None = None,
    ) -> pd.DataFrame:
        """Train the model and return one row of metrics per epoch."""

        set_global_seed(
            self.config.seed,
            deterministic=self.config.deterministic,
        )
        if self.config.monitor.startswith("validation_") and validation_loader is None:
            raise ValueError(
                f"monitor={self.config.monitor!r} requires a validation loader"
            )

        records: list[dict[str, float | int]] = []
        best_state: dict[str, Tensor] | None = None
        best_value = math.inf
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            beta_kl = _ramped_weight(
                self.config.beta_kl,
                epoch,
                self.config.kl_warmup_epochs,
            )
            lambda_adversarial = _ramped_weight(
                self.config.lambda_adversarial,
                epoch,
                self.config.adversarial_warmup_epochs,
            )

            train_metrics = self._train_epoch(
                train_loader,
                beta_kl=beta_kl,
                lambda_adversarial=lambda_adversarial,
            )
            record: dict[str, float | int] = {
                "epoch": epoch,
                "beta_kl": beta_kl,
                "lambda_adversarial": lambda_adversarial,
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }

            if validation_loader is not None:
                validation_metrics = self._evaluate(
                    validation_loader,
                    beta_kl=beta_kl,
                    lambda_adversarial=lambda_adversarial,
                )
                record.update(
                    {
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    }
                )

            monitor_value = float(record[self.config.monitor])
            if not math.isfinite(monitor_value):
                raise FloatingPointError(
                    f"Non-finite monitored metric at epoch {epoch}: "
                    f"{self.config.monitor}={monitor_value}"
                )

            if monitor_value < best_value - self.config.early_stopping_min_delta:
                best_value = monitor_value
                self.best_epoch_ = epoch
                self.best_monitor_value_ = monitor_value
                epochs_without_improvement = 0
                if self.config.restore_best_weights:
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
            else:
                epochs_without_improvement += 1

            records.append(record)
            self._log_epoch(record, validation_loader is not None)

            if (
                self.config.early_stopping_patience is not None
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None and self.config.restore_best_weights:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.history_ = pd.DataFrame.from_records(records)
        return self.history_.copy()

    def _train_epoch(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        *,
        beta_kl: float,
        lambda_adversarial: float,
    ) -> dict[str, float]:
        self.model.train()
        metrics = _MetricAccumulator()

        for x, domain in loader:
            x = x.to(self.device, non_blocking=True)
            domain = domain.to(self.device, dtype=torch.long, non_blocking=True)

            discriminator_loss = self._update_discriminator(x, domain)
            terms, domain_logits = self._update_autoencoder(
                x,
                domain,
                beta_kl=beta_kl,
                lambda_adversarial=lambda_adversarial,
            )

            predictions = domain_logits.argmax(dim=1)
            domain_accuracy = (predictions == domain).float().mean().item()
            metrics.add(
                x.shape[0],
                reconstruction=terms.reconstruction.detach().item(),
                kl=terms.kl.detach().item(),
                sparsity=terms.sparsity.detach().item(),
                domain_loss=terms.domain.detach().item(),
                domain_accuracy=domain_accuracy,
                discriminator_loss=discriminator_loss,
                autoencoder_objective=terms.autoencoder_objective.detach().item(),
            )

        return metrics.means()

    def _update_discriminator(self, x: Tensor, domain: Tensor) -> float:
        discriminator_loss_value = math.nan
        self._set_discriminator_requires_grad(True)

        for _ in range(self.config.discriminator_steps):
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                mu, logvar = self.model.encode(x, domain)
                z = self.model.reparameterize(mu, logvar)

            logits = self.model.discriminate(z)
            loss = F.cross_entropy(
                logits,
                domain,
                weight=self.domain_class_weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite discriminator loss")

            loss.backward()
            self._clip(self.model.discriminator_parameters())
            self.discriminator_optimizer.step()
            discriminator_loss_value = float(loss.detach().item())

        return discriminator_loss_value

    def _update_autoencoder(
        self,
        x: Tensor,
        domain: Tensor,
        *,
        beta_kl: float,
        lambda_adversarial: float,
    ) -> tuple[LossTerms, Tensor]:
        self._set_discriminator_requires_grad(False)
        try:
            self.autoencoder_optimizer.zero_grad(set_to_none=True)
            output = self.model(x, domain, sample_latent=True)
            terms = _calculate_loss_terms(
                output,
                x,
                domain,
                reconstruction_weight=self.config.reconstruction_weight,
                beta_kl=beta_kl,
                lambda_l1=self.config.lambda_l1,
                lambda_adversarial=lambda_adversarial,
                domain_class_weights=self.domain_class_weights,
            )
            if not torch.isfinite(terms.autoencoder_objective):
                raise FloatingPointError("Non-finite autoencoder objective")

            terms.autoencoder_objective.backward()
            self._clip(self.model.autoencoder_parameters())
            self.autoencoder_optimizer.step()
            return terms, output.domain_logits.detach()
        finally:
            self._set_discriminator_requires_grad(True)

    @torch.inference_mode()
    def _evaluate(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        *,
        beta_kl: float,
        lambda_adversarial: float,
    ) -> dict[str, float]:
        self.model.eval()
        metrics = _MetricAccumulator()

        for x, domain in loader:
            x = x.to(self.device, non_blocking=True)
            domain = domain.to(self.device, dtype=torch.long, non_blocking=True)
            output = self.model(x, domain, sample_latent=False)
            terms = _calculate_loss_terms(
                output,
                x,
                domain,
                reconstruction_weight=self.config.reconstruction_weight,
                beta_kl=beta_kl,
                lambda_l1=self.config.lambda_l1,
                lambda_adversarial=lambda_adversarial,
                domain_class_weights=self.domain_class_weights,
            )
            predictions = output.domain_logits.argmax(dim=1)
            metrics.add(
                x.shape[0],
                reconstruction=terms.reconstruction.item(),
                kl=terms.kl.item(),
                sparsity=terms.sparsity.item(),
                domain_loss=terms.domain.item(),
                domain_accuracy=(predictions == domain).float().mean().item(),
                autoencoder_objective=terms.autoencoder_objective.item(),
            )

        return metrics.means()

    def _prepare_class_weights(
        self,
        weights: Sequence[float] | np.ndarray | Tensor | None,
    ) -> Tensor | None:
        if weights is None:
            return None

        tensor = torch.as_tensor(weights, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.numel() != self.model.n_domains:
            raise ValueError("domain_class_weights must contain one value per domain")
        if torch.any(tensor <= 0):
            raise ValueError("domain_class_weights must be positive")
        return tensor.to(self.device)

    def _set_discriminator_requires_grad(self, enabled: bool) -> None:
        for parameter in self.model.discriminator_parameters():
            parameter.requires_grad_(enabled)

    def _clip(self, parameters: Iterable[nn.Parameter]) -> None:
        if self.config.gradient_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            list(parameters),
            max_norm=self.config.gradient_clip_norm,
        )

    def _log_epoch(
        self,
        record: dict[str, float | int],
        has_validation: bool,
    ) -> None:
        epoch = int(record["epoch"])
        if not self.config.log_every:
            return
        if (
            epoch != 1
            and epoch % self.config.log_every != 0
            and epoch != self.config.epochs
        ):
            return

        message = (
            "epoch=%04d train_recon=%.5f train_domain_acc=%.3f",
            epoch,
            record["train_reconstruction"],
            record["train_domain_accuracy"],
        )
        if has_validation:
            logger.info(
                message[0] + " val_recon=%.5f val_domain_acc=%.3f",
                *message[1:],
                record["validation_reconstruction"],
                record["validation_domain_accuracy"],
            )
        else:
            logger.info(*message)
