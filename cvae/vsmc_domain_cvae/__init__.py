"""Domain-adversarial CVAE tools for VSMC proteome integration."""

from .common import set_global_seed
from .diann import diann_report_to_protein_matrix
from .inference import encode_proteomes, reconstruct_or_translate_proteomes
from .model import CVAEConfig, CVAEOutput, DomainAdversarialCVAE
from .network import NetworkFeatureSelectionResult, select_conserved_network_proteins
from .persistence import LoadedCVAEBundle, load_cvae_bundle, save_cvae_bundle
from .preprocessing import DomainDataBundle, ProteomeStandardizer, prepare_domain_data
from .training import CVAETrainer, TrainingConfig

__version__ = "0.2.0"

__all__ = [
    "CVAEConfig",
    "CVAEOutput",
    "CVAETrainer",
    "DomainAdversarialCVAE",
    "DomainDataBundle",
    "LoadedCVAEBundle",
    "NetworkFeatureSelectionResult",
    "ProteomeStandardizer",
    "TrainingConfig",
    "diann_report_to_protein_matrix",
    "encode_proteomes",
    "load_cvae_bundle",
    "prepare_domain_data",
    "reconstruct_or_translate_proteomes",
    "save_cvae_bundle",
    "select_conserved_network_proteins",
    "set_global_seed",
]
