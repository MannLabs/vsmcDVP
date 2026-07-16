import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from vsmc_domain_cvae import (
    CVAEConfig,
    CVAETrainer,
    DomainAdversarialCVAE,
    TrainingConfig,
    diann_report_to_protein_matrix,
    encode_proteomes,
    load_cvae_bundle,
    prepare_domain_data,
    reconstruct_or_translate_proteomes,
    save_cvae_bundle,
    select_conserved_network_proteins,
)
from vsmc_domain_cvae.network import (
    _rowwise_network_profile_similarity,
    _spearman_correlation_matrix,
)


torch.set_num_threads(1)


class TestVSMCDomainCVAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(7)
        n_ref, n_nbr, p = 36, 32, 24
        loadings = rng.normal(size=(3, p))
        ref_latent = rng.normal(size=(n_ref, 3))
        nbr_latent = rng.normal(size=(n_nbr, 3))
        ref = ref_latent @ loadings + 0.2 * rng.normal(size=(n_ref, p))
        nbr = nbr_latent @ loadings + 0.2 * rng.normal(size=(n_nbr, p))
        nbr += np.linspace(-0.5, 0.5, p)[None, :]
        # Make a few proteins domain-specific/noisy to exercise network filtering.
        nbr[:, -4:] = rng.normal(size=(n_nbr, 4))
        cls.proteins = [f"P{i:04d}" for i in range(p)]
        cls.reference = pd.DataFrame(
            ref, index=[f"R{i:03d}" for i in range(n_ref)], columns=cls.proteins
        )
        cls.neighborhood = pd.DataFrame(
            nbr, index=[f"N{i:03d}" for i in range(n_nbr)], columns=cls.proteins
        )
        cls.reference.iloc[0, 0] = np.nan
        cls.neighborhood.iloc[1, 1] = np.nan

    def test_blockwise_network_math_matches_scipy(self) -> None:
        rng = np.random.default_rng(101)
        first = rng.normal(size=(17, 9))
        second = rng.normal(size=(14, 9))
        corr_first = _spearman_correlation_matrix(
            first, block_size=4, dtype=np.dtype("float64")
        )
        corr_second = _spearman_correlation_matrix(
            second, block_size=3, dtype=np.dtype("float64")
        )
        np.testing.assert_allclose(
            corr_first, spearmanr(first, axis=0).statistic, atol=1e-12
        )
        observed = _rowwise_network_profile_similarity(
            corr_first, corr_second, block_size=3
        )
        expected = np.asarray(
            [
                spearmanr(
                    np.delete(corr_first[i], i),
                    np.delete(corr_second[i], i),
                ).statistic
                for i in range(corr_first.shape[0])
            ]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_diann_report_reshape(self) -> None:
        report = pd.DataFrame(
            {
                "Run": ["A", "A", "B", "B"],
                "Protein.Group": ["P1", "P1", "P1", "P2"],
                "PG.MaxLFQ": [4.0, 16.0, 8.0, 32.0],
                "PG.Q.Value": [0.001, 0.002, 0.003, 0.02],
            }
        )
        matrix = diann_report_to_protein_matrix(report, max_q_value=0.01)

        self.assertEqual(matrix.index.tolist(), ["A", "B"])
        self.assertEqual(matrix.columns.tolist(), ["P1"])
        self.assertAlmostEqual(matrix.loc["A", "P1"], np.log2(10.0))
        self.assertAlmostEqual(matrix.loc["B", "P1"], 3.0)

    def test_network_selection(self) -> None:
        result = select_conserved_network_proteins(
            self.reference,
            self.neighborhood,
            threshold=0.1,
            correlation_block_size=8,
            similarity_block_size=7,
        )
        self.assertEqual(result.n_shared, len(self.proteins))
        self.assertGreater(result.n_selected, 0)
        self.assertTrue(set(result.selected_proteins).issubset(self.proteins))
        self.assertIn("network_similarity", result.metrics.columns)

    def test_train_infer_save_load(self) -> None:
        bundle = prepare_domain_data(
            self.reference,
            self.neighborhood,
            proteins=self.proteins[:16],
            validation_fraction=0.2,
            batch_size=16,
            seed=3,
        )
        model_config = CVAEConfig(
            input_dim=16,
            latent_dim=4,
            encoder_hidden=(24, 12),
            decoder_hidden=(12, 24),
            discriminator_hidden=(10,),
            dropout=0.0,
        )
        model = DomainAdversarialCVAE(model_config)
        train_config = TrainingConfig(
            epochs=3,
            beta_kl=0.01,
            lambda_adversarial=0.05,
            lambda_l1=1e-4,
            kl_warmup_epochs=0,
            adversarial_warmup_epochs=0,
            early_stopping_patience=None,
            log_every=0,
            seed=3,
            device="cpu",
        )
        trainer = CVAETrainer(
            model,
            train_config,
            domain_class_weights=bundle.domain_class_weights,
        )
        history = trainer.fit(bundle.train_loader, bundle.validation_loader)
        self.assertEqual(len(history), 3)
        self.assertTrue(np.isfinite(history["train_reconstruction"]).all())

        ref_subset = self.reference.loc[:, self.proteins[:16]].iloc[:5]
        embedding = encode_proteomes(
            model,
            ref_subset,
            domains=0,
            transformer=bundle.transformer,
            feature_names=bundle.feature_names,
            device="cpu",
        )
        self.assertEqual(embedding.shape, (5, 5))  # domain index + four z values

        translated = reconstruct_or_translate_proteomes(
            model,
            ref_subset,
            source_domains=0,
            target_domains=1,
            transformer=bundle.transformer,
            feature_names=bundle.feature_names,
            device="cpu",
        )
        self.assertEqual(translated.shape, (5, 16))
        self.assertTrue(np.isfinite(translated.to_numpy()).all())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_cvae_bundle(
                path,
                model,
                feature_names=bundle.feature_names,
                domain_names=bundle.domain_names,
                transformer=bundle.transformer,
                training_config=train_config,
                history=history,
                extra={"purpose": "unit_test"},
            )
            loaded = load_cvae_bundle(path)
            self.assertEqual(loaded.feature_names, bundle.feature_names)
            self.assertEqual(loaded.extra["purpose"], "unit_test")
            loaded_embedding = encode_proteomes(
                loaded.model,
                ref_subset,
                domains=0,
                transformer=loaded.transformer,
                feature_names=loaded.feature_names,
                device="cpu",
            )
            np.testing.assert_allclose(
                embedding.filter(like="z_").to_numpy(),
                loaded_embedding.filter(like="z_").to_numpy(),
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
