"""Command-line pipeline for feature selection, training, and export."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .common import set_global_seed
from .inference import encode_proteomes
from .model import CVAEConfig, DomainAdversarialCVAE
from .network import NetworkFeatureSelectionResult, select_conserved_network_proteins
from .persistence import save_cvae_bundle
from .preprocessing import DomainDataBundle, prepare_domain_data
from .training import CVAETrainer, TrainingConfig

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select proteins with conserved Spearman co-expression networks and "
            "train a domain-adversarial conditional VAE."
        )
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--neighborhood", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--separator",
        default="\t",
        help="Input delimiter. The default is a tab character.",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--missing",
        choices=("error", "median", "zero"),
        default="median",
        help="Missing-value strategy for network estimation and model input.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--correlation-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--correlation-block-size", type=int, default=512)
    parser.add_argument("--similarity-block-size", type=int, default=128)
    parser.add_argument("--expected-shared", type=int)
    parser.add_argument("--expected-selected", type=int)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--lambda-adversarial", type=float, default=0.1)
    parser.add_argument("--lambda-l1", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--device",
        help="PyTorch device, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable training-set per-protein z-scoring.",
    )
    parser.add_argument(
        "--no-balance-domains",
        action="store_true",
        help="Disable inverse-frequency sampling of the training domains.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def read_matrix(path: Path, separator: str) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep=separator, index_col=0, low_memory=False)
    matrix.index = matrix.index.astype(str)
    return matrix


def _check_expected_count(name: str, observed: int, expected: int | None) -> None:
    if expected is not None and observed != expected:
        raise RuntimeError(f"Expected {expected} {name}, observed {observed}")


def _write_feature_selection(
    selection: NetworkFeatureSelectionResult,
    output_dir: Path,
) -> None:
    selection.save_metrics(output_dir / "network_feature_selection.tsv")
    pd.Series(selection.selected_proteins, name="protein").to_csv(
        output_dir / "selected_proteins.txt",
        index=False,
        header=False,
    )


def _write_embeddings(
    model: DomainAdversarialCVAE,
    data: DomainDataBundle,
    reference: pd.DataFrame,
    neighborhood: pd.DataFrame,
    *,
    device: str,
    output_path: Path,
) -> None:
    frames: list[pd.DataFrame] = []
    for matrix, domain_index, domain_name in (
        (reference, 0, data.domain_names[0]),
        (neighborhood, 1, data.domain_names[1]),
    ):
        embedding = encode_proteomes(
            model,
            matrix,
            domains=domain_index,
            transformer=data.transformer,
            feature_names=data.feature_names,
            device=device,
        )
        embedding.insert(1, "domain", domain_name)
        embedding.insert(0, "sample_id", embedding.index.astype(str))
        frames.append(embedding.reset_index(drop=True))

    pd.concat(frames, ignore_index=True).to_csv(
        output_path,
        sep="\t",
        index=False,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading input matrices")
    reference = read_matrix(args.reference, args.separator)
    neighborhood = read_matrix(args.neighborhood, args.separator)

    logger.info("Selecting network-conserved proteins")
    selection = select_conserved_network_proteins(
        reference,
        neighborhood,
        threshold=args.threshold,
        missing=args.missing,
        correlation_block_size=args.correlation_block_size,
        similarity_block_size=args.similarity_block_size,
        correlation_dtype=args.correlation_dtype,
        cache_dir=args.cache_dir,
    )
    _check_expected_count(
        "shared proteins",
        selection.n_shared,
        args.expected_shared,
    )
    _check_expected_count(
        "network-conserved proteins",
        selection.n_selected,
        args.expected_selected,
    )
    if not selection.selected_proteins:
        raise RuntimeError("No proteins passed the network-conservation threshold")

    _write_feature_selection(selection, args.output_dir)

    data = prepare_domain_data(
        reference,
        neighborhood,
        proteins=selection.selected_proteins,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
        impute=args.missing,
        standardize=not args.no_standardize,
        balance_training_domains=not args.no_balance_domains,
        num_workers=args.num_workers,
    )
    data.sample_table.to_csv(
        args.output_dir / "sample_split_metadata.tsv",
        sep="\t",
        index=False,
    )

    set_global_seed(args.seed)
    model = DomainAdversarialCVAE(
        CVAEConfig(
            input_dim=len(data.feature_names),
            n_domains=len(data.domain_names),
            latent_dim=args.latent_dim,
            encoder_hidden=(512, 256),
            decoder_hidden=(256, 512),
            discriminator_hidden=(128, 64),
            activation="gelu",
            dropout=0.1,
            layer_norm=True,
        )
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        beta_kl=args.beta_kl,
        lambda_adversarial=args.lambda_adversarial,
        lambda_l1=args.lambda_l1,
        kl_warmup_epochs=args.kl_warmup_epochs,
        adversarial_warmup_epochs=args.adversarial_warmup_epochs,
        seed=args.seed,
        device=args.device,
        monitor=(
            "validation_reconstruction"
            if data.validation_loader is not None
            else "train_reconstruction"
        ),
        log_every=args.log_every,
    )

    logger.info("Training CVAE")
    trainer = CVAETrainer(
        model,
        training_config,
        domain_class_weights=data.domain_class_weights,
    )
    history = trainer.fit(data.train_loader, data.validation_loader)
    history.to_csv(
        args.output_dir / "training_history.tsv",
        sep="\t",
        index=False,
    )

    _write_embeddings(
        model,
        data,
        reference,
        neighborhood,
        device=str(trainer.device),
        output_path=args.output_dir / "latent_embeddings.tsv",
    )

    save_cvae_bundle(
        args.output_dir / "vsmc_domain_cvae.pt",
        model,
        feature_names=data.feature_names,
        domain_names=data.domain_names,
        transformer=data.transformer,
        training_config=training_config,
        history=history,
        extra={
            "network_threshold": args.threshold,
            "correlation_dtype": args.correlation_dtype,
            "n_shared": selection.n_shared,
            "n_network_eligible": selection.n_network_eligible,
            "n_selected": selection.n_selected,
            "best_epoch": trainer.best_epoch_,
            "best_monitor_value": trainer.best_monitor_value_,
            "torch_version": torch.__version__,
        },
    )

    summary: dict[str, object] = {
        "reference_profiles": int(reference.shape[0]),
        "neighborhood_profiles": int(neighborhood.shape[0]),
        "shared_proteins": selection.n_shared,
        "network_eligible_proteins": selection.n_network_eligible,
        "selected_proteins": selection.n_selected,
        "network_similarity_threshold": args.threshold,
        "correlation_dtype": args.correlation_dtype,
        "latent_dim": args.latent_dim,
        "epochs_completed": int(len(history)),
        "best_epoch": trainer.best_epoch_,
        "best_monitor_value": trainer.best_monitor_value_,
        "device": str(trainer.device),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
