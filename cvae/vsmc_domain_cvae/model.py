"""Conditional VAE and latent-domain discriminator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import ActivationName


def _activation(name: ActivationName) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "silu":
            return nn.SiLU()
        case "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1)
        case _:
            raise ValueError(f"Unsupported activation: {name!r}")


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: ActivationName,
    dropout: float,
    layer_norm: bool,
) -> tuple[nn.Module, int]:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    layers: list[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("All hidden dimensions must be positive")

        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_activation(activation))
        if dropout:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    network = nn.Sequential(*layers) if layers else nn.Identity()
    return network, current_dim


@dataclass(frozen=True)
class CVAEConfig:
    input_dim: int
    n_domains: int = 2
    latent_dim: int = 16
    encoder_hidden: tuple[int, ...] = (512, 256)
    decoder_hidden: tuple[int, ...] = (256, 512)
    discriminator_hidden: tuple[int, ...] = (128, 64)
    activation: ActivationName = "gelu"
    dropout: float = 0.1
    layer_norm: bool = True
    logvar_min: float = -12.0
    logvar_max: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_hidden", tuple(self.encoder_hidden))
        object.__setattr__(self, "decoder_hidden", tuple(self.decoder_hidden))
        object.__setattr__(
            self,
            "discriminator_hidden",
            tuple(self.discriminator_hidden),
        )

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.n_domains < 2:
            raise ValueError("n_domains must be at least 2")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")


class CVAEOutput(NamedTuple):
    reconstruction: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor
    domain_logits: Tensor


class DomainAdversarialCVAE(nn.Module):
    """Conditional VAE with an adversarial latent-domain classifier."""

    def __init__(self, config: CVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder_backbone, encoder_output_dim = _make_mlp(
            config.input_dim + config.n_domains,
            config.encoder_hidden,
            activation=config.activation,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.encoder_mu = nn.Linear(encoder_output_dim, config.latent_dim)
        self.encoder_logvar = nn.Linear(encoder_output_dim, config.latent_dim)

        self.decoder_backbone, decoder_output_dim = _make_mlp(
            config.latent_dim + config.n_domains,
            config.decoder_hidden,
            activation=config.activation,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.decoder_output = nn.Linear(decoder_output_dim, config.input_dim)

        self.discriminator_backbone, discriminator_output_dim = _make_mlp(
            config.latent_dim,
            config.discriminator_hidden,
            activation=config.activation,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
        )
        self.discriminator_output = nn.Linear(
            discriminator_output_dim,
            config.n_domains,
        )

        self.reset_parameters()

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    @property
    def n_domains(self) -> int:
        return self.config.n_domains

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: Tensor, domain: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_batch(x, domain, expected_width=self.input_dim, name="x")
        conditioned = torch.cat((x, self._one_hot_domain(domain, x.dtype)), dim=1)
        hidden = self.encoder_backbone(conditioned)
        mu = self.encoder_mu(hidden)
        logvar = self.encoder_logvar(hidden).clamp(
            min=self.config.logvar_min,
            max=self.config.logvar_max,
        )
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mu + standard_deviation * noise

    def decode(self, z: Tensor, domain: Tensor) -> Tensor:
        self._validate_batch(z, domain, expected_width=self.latent_dim, name="z")
        conditioned = torch.cat((z, self._one_hot_domain(domain, z.dtype)), dim=1)
        hidden = self.decoder_backbone(conditioned)
        return self.decoder_output(hidden)

    def discriminate(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape (batch, {self.latent_dim})")
        hidden = self.discriminator_backbone(z)
        return self.discriminator_output(hidden)

    def forward(
        self,
        x: Tensor,
        domain: Tensor,
        *,
        sample_latent: bool = True,
    ) -> CVAEOutput:
        mu, logvar = self.encode(x, domain)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        reconstruction = self.decode(z, domain)
        domain_logits = self.discriminate(z)
        return CVAEOutput(reconstruction, mu, logvar, z, domain_logits)

    def autoencoder_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.encoder_backbone,
            self.encoder_mu,
            self.encoder_logvar,
            self.decoder_backbone,
            self.decoder_output,
        )
        for module in modules:
            yield from module.parameters()

    def discriminator_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.discriminator_backbone.parameters()
        yield from self.discriminator_output.parameters()

    def _one_hot_domain(self, domain: Tensor, dtype: torch.dtype) -> Tensor:
        if domain.ndim != 1:
            raise ValueError("domain labels must be one-dimensional")
        if domain.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("domain labels must have an integer dtype")
        if domain.numel() and (
            int(domain.min().item()) < 0
            or int(domain.max().item()) >= self.n_domains
        ):
            raise ValueError("domain label is outside the configured range")

        return F.one_hot(
            domain.to(torch.int64),
            num_classes=self.n_domains,
        ).to(dtype=dtype)

    @staticmethod
    def _validate_batch(
        values: Tensor,
        domain: Tensor,
        *,
        expected_width: int,
        name: str,
    ) -> None:
        if values.ndim != 2 or values.shape[1] != expected_width:
            raise ValueError(f"{name} must have shape (batch, {expected_width})")
        if values.shape[0] != domain.shape[0]:
            raise ValueError(f"{name} and domain have different batch sizes")
