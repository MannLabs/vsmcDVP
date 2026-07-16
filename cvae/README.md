# VSMC domain-adversarial CVAE

PyTorch implementation of the VSMC reference-to-neighborhood integration workflow:

1. intersect the two proteome matrices on shared proteins;
2. retain proteins with conserved Spearman co-expression neighborhoods;
3. prepare a domain-stratified training and validation split;
4. train a conditional VAE with an adversarial domain discriminator; and
5. export latent embeddings, reconstructions, cross-domain translations, and a self-contained checkpoint.

Input abundance matrices use profiles in rows and protein IDs in columns.

## Package layout

```text
vsmc_domain_cvae/
├── __init__.py       # stable public API
├── __main__.py       # python -m vsmc_domain_cvae entry point
├── common.py         # shared types, validation, seeding, device handling
├── diann.py          # DIA-NN long-report reshaping
├── network.py        # co-expression matrices and feature selection
├── preprocessing.py  # imputation, scaling, train/validation loaders
├── model.py          # encoder, decoder, and discriminator
├── training.py       # minimax losses and alternating optimization
├── inference.py      # embedding, reconstruction, and translation
├── persistence.py    # model and preprocessing persistence
└── cli.py            # command-line pipeline
```

`run_vsmc_domain_cvae.py` is a thin command-line wrapper. The same pipeline is available after installation as `vsmc-domain-cvae` or through `python -m vsmc_domain_cvae`. The package root re-exports the supported public API, so existing client imports do not need to change.

## Installation

Install from the project directory:

```bash
python -m pip install .
```

For a source checkout without installation:

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or newer is required.

## Command-line analysis

```bash
python -m vsmc_domain_cvae \
  --reference merged_cellular_secretome.tsv \
  --neighborhood filtered_dvp_neighborhood.tsv \
  --output-dir cvae_results \
  --threshold 0.3 \
  --expected-shared 4727 \
  --expected-selected 1015 \
  --cache-dir /path/to/fast_scratch
```

The two input files are tab-separated matrices whose first column contains sample or profile IDs. `--expected-shared` and `--expected-selected` are optional assertions for reproducing the reported 4,727 shared and 1,015 selected proteins.

The pipeline writes:

- `network_feature_selection.tsv`
- `selected_proteins.txt`
- `sample_split_metadata.tsv`
- `training_history.tsv`
- `latent_embeddings.tsv`
- `vsmc_domain_cvae.pt`
- `run_summary.json`

## Python API

```python
import pandas as pd

from vsmc_domain_cvae import (
    CVAEConfig,
    CVAETrainer,
    DomainAdversarialCVAE,
    TrainingConfig,
    encode_proteomes,
    prepare_domain_data,
    save_cvae_bundle,
    select_conserved_network_proteins,
)

reference = pd.read_csv("merged_cellular_secretome.tsv", sep="\t", index_col=0)
neighborhood = pd.read_csv(
    "filtered_dvp_neighborhood.tsv",
    sep="\t",
    index_col=0,
)

selection = select_conserved_network_proteins(
    reference,
    neighborhood,
    threshold=0.3,
    missing="median",
    cache_dir="/path/to/fast_scratch",
)

data = prepare_domain_data(
    reference,
    neighborhood,
    proteins=selection.selected_proteins,
    validation_fraction=0.2,
    batch_size=64,
    seed=13,
)

model = DomainAdversarialCVAE(
    CVAEConfig(
        input_dim=len(data.feature_names),
        latent_dim=16,
        encoder_hidden=(512, 256),
        decoder_hidden=(256, 512),
        discriminator_hidden=(128, 64),
    )
)

training = TrainingConfig(
    epochs=200,
    beta_kl=0.01,
    lambda_adversarial=0.1,
    lambda_l1=1e-4,
    device="cuda",
)
trainer = CVAETrainer(
    model,
    training,
    domain_class_weights=data.domain_class_weights,
)
history = trainer.fit(data.train_loader, data.validation_loader)

reference_z = encode_proteomes(
    model,
    reference,
    domains=0,
    transformer=data.transformer,
    feature_names=data.feature_names,
)

save_cvae_bundle(
    "vsmc_domain_cvae.pt",
    model,
    feature_names=data.feature_names,
    domain_names=data.domain_names,
    transformer=data.transformer,
    training_config=training,
    history=history,
)
```

## Network feature selection

For each shared protein, `network.py` builds a Spearman co-expression vector independently in the reference and neighborhood matrices. It then calculates the Spearman correlation between those two edge vectors after removing the self-edge. Retention uses the strict criterion

```text
network_similarity > threshold
```

The default threshold is `0.3`. Missing values are median-imputed separately within each domain for network estimation unless `missing="error"` or `missing="zero"` is selected. Proteins that are all-missing, non-finite after imputation, or effectively constant in either domain are excluded and annotated in the audit table.

At 4,727 proteins, two dense float32 correlation matrices require roughly 171 MiB. Supplying `cache_dir` stores them as temporary memory maps.

## Training objective

The autoencoder minimizes

```text
reconstruction_weight * MSE
+ beta_kl * KL(q(z|x,c) || N(0,I))
+ lambda_l1 * mean(abs(z))
- lambda_adversarial * cross_entropy(domain_logits, domain)
```

The discriminator is updated separately to minimize domain cross-entropy on a detached latent sample. During the autoencoder update, discriminator parameters are frozen while gradients still pass through the discriminator to the encoder.

The loss coefficients in `TrainingConfig` are tunable defaults; they are not inferred from the methods description.

## DIA-NN utility

```python
from vsmc_domain_cvae import diann_report_to_protein_matrix

matrix = diann_report_to_protein_matrix(
    "report.tsv",
    sample_col="Run",
    protein_col="Protein.Group",
    quantity_col="PG.MaxLFQ",
    q_value_col="PG.Q.Value",
    max_q_value=0.01,
)
```

The function keeps DIA-NN protein-group identifiers as supplied. Protein inference, contaminant filtering, normalization, and other study-specific processing remain upstream responsibilities.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover DIA-NN reshaping, the blockwise Spearman implementation, network-conserved feature selection, model training, latent inference, conditional decoding, and checkpoint round-tripping.
