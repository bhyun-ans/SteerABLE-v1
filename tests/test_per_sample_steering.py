# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-sample epitope steering must keep diffusion samples independent.

Batching the N_sample axis is only safe if one sample's reward can never reach
another sample's steered embedding.  Two things could break that, and both are
checked here:

  1. the reward reduction -- collapsing the sample axis to a scalar makes every
     sample share one gradient (the chunk-reward-averaging failure);
  2. the RMS normaliser -- a global RMS couples the samples through the
     denominator even when the gradients themselves are separate.

The tests also pin the chunk_size=1 behaviour, so the batched path provably
reduces to the serial one when there is only one sample in the chunk.
"""

import unittest

import torch

from protenix.model.steering import (
    contact_epitope_reward,
    epitope_clash_reward,
    rms_normalize,
)


def _relp_feature(conditioning, n_token: int) -> torch.Tensor:
    """Build the one-hot relative-position tensor DiffusionConditioning expects."""
    feats = {
        "asym_id": torch.zeros(n_token, dtype=torch.long),
        "residue_index": torch.arange(n_token),
        "entity_id": torch.zeros(n_token, dtype=torch.long),
        "token_index": torch.arange(n_token),
        "sym_id": torch.zeros(n_token, dtype=torch.long),
    }
    return conditioning.relpe.generate_relp(feats)["relp"]


def _toy_masks(n_atom: int, n_epitope: int = 2, n_partner: int = 4):
    """One Ag chain group: `n_epitope` single-atom residues plus a partner mask."""
    residue_groups = [torch.tensor([i], dtype=torch.long) for i in range(n_epitope)]
    partner_mask = torch.zeros(n_atom, dtype=torch.bool)
    partner_mask[n_atom - n_partner :] = True
    return [(residue_groups, partner_mask)]


class TestRewardSampleIsolation(unittest.TestCase):
    """Sample j's reward must not depend on sample i's coordinates."""

    def setUp(self):
        torch.manual_seed(0)
        self.n_sample, self.n_atom = 4, 12
        self.mask_pairs = _toy_masks(self.n_atom)
        self.ref_element = torch.zeros(self.n_atom, 128)
        self.ref_element[:, 6] = 1.0  # carbon

    def test_contact_reward_is_block_diagonal_in_samples(self):
        coords = torch.randn(self.n_sample, self.n_atom, 3, requires_grad=True)
        r = contact_epitope_reward(
            coords=coords, mask_pairs=self.mask_pairs, per_sample=True
        )
        self.assertEqual(r.shape, (self.n_sample,))

        for j in range(self.n_sample):
            grad = torch.autograd.grad(r[j], coords, retain_graph=True)[0]
            for i in range(self.n_sample):
                if i == j:
                    self.assertGreater(
                        grad[i].abs().sum().item(),
                        0.0,
                        f"reward of sample {j} does not depend on its own coords",
                    )
                else:
                    self.assertEqual(
                        grad[i].abs().sum().item(),
                        0.0,
                        f"reward of sample {j} leaked into sample {i}",
                    )

    def test_clash_reward_is_block_diagonal_in_samples(self):
        coords = torch.randn(self.n_sample, self.n_atom, 3, requires_grad=True) * 0.5
        r = epitope_clash_reward(
            coords=coords,
            mask_pairs=self.mask_pairs,
            ref_element=self.ref_element,
            per_sample=True,
        )
        self.assertEqual(r.shape, (self.n_sample,))
        for j in range(self.n_sample):
            grad = torch.autograd.grad(r[j], coords, retain_graph=True)[0]
            for i in range(self.n_sample):
                if i != j:
                    self.assertEqual(
                        grad[i].abs().sum().item(),
                        0.0,
                        f"clash reward of sample {j} leaked into sample {i}",
                    )

    def test_summed_reward_gradient_equals_serial_gradient(self):
        """grad of sum_i r_i wrt coords == what each sample gets run alone.

        This is the identity the batched sampler relies on:
            d(sum_i r_i)/d x_j = d r_j / d x_j
        """
        coords = torch.randn(self.n_sample, self.n_atom, 3)

        batched = coords.clone().requires_grad_(True)
        r = contact_epitope_reward(
            coords=batched, mask_pairs=self.mask_pairs, per_sample=True
        )
        g_batched = torch.autograd.grad(r.sum(), batched)[0]

        for j in range(self.n_sample):
            solo = coords[j : j + 1].clone().requires_grad_(True)
            r_solo = contact_epitope_reward(
                coords=solo, mask_pairs=self.mask_pairs, per_sample=True
            )
            g_solo = torch.autograd.grad(r_solo.sum(), solo)[0]
            torch.testing.assert_close(g_batched[j], g_solo[0], rtol=0, atol=0)


class TestRewardReductionCompatibility(unittest.TestCase):
    """per_sample=True with one sample must reproduce the old scalar reward."""

    def setUp(self):
        torch.manual_seed(1)
        self.n_atom = 12
        self.mask_pairs = _toy_masks(self.n_atom)
        self.ref_element = torch.zeros(self.n_atom, 128)
        self.ref_element[:, 6] = 1.0

    def test_single_sample_matches_scalar_path(self):
        coords = torch.randn(1, self.n_atom, 3)
        scalar = contact_epitope_reward(coords=coords, mask_pairs=self.mask_pairs)
        vector = contact_epitope_reward(
            coords=coords, mask_pairs=self.mask_pairs, per_sample=True
        )
        self.assertEqual(vector.shape, (1,))
        torch.testing.assert_close(vector.sum(), scalar, rtol=0, atol=0)

    def test_scalar_path_is_mean_over_samples(self):
        """The old scalar reward is the mean; keep that for any caller still using it."""
        coords = torch.randn(3, self.n_atom, 3)
        scalar = contact_epitope_reward(coords=coords, mask_pairs=self.mask_pairs)
        vector = contact_epitope_reward(
            coords=coords, mask_pairs=self.mask_pairs, per_sample=True
        )
        torch.testing.assert_close(vector.mean(), scalar, rtol=1e-6, atol=1e-6)

    def test_empty_mask_pairs_shape(self):
        coords = torch.randn(3, self.n_atom, 3)
        self.assertEqual(
            contact_epitope_reward(coords=coords, mask_pairs=[], per_sample=True).shape,
            (3,),
        )
        self.assertEqual(
            contact_epitope_reward(coords=coords, mask_pairs=[]).shape, ()
        )
        self.assertEqual(
            epitope_clash_reward(
                coords=coords,
                mask_pairs=[],
                ref_element=self.ref_element,
                per_sample=True,
            ).shape,
            (3,),
        )


class TestRmsNormalizePerSample(unittest.TestCase):
    """The normaliser must not couple samples through its denominator."""

    def test_per_sample_rms_is_independent(self):
        torch.manual_seed(2)
        x = torch.randn(4, 7, 5)
        out = rms_normalize(x, sample_dim=-3)

        # Each slice is normalised by its own RMS.
        for i in range(x.shape[0]):
            solo = rms_normalize(x[i : i + 1], sample_dim=-3)
            torch.testing.assert_close(out[i], solo[0], rtol=0, atol=0)

        # Scaling one sample must leave the others bit-identical.
        y = x.clone()
        y[0] *= 100.0
        out_y = rms_normalize(y, sample_dim=-3)
        torch.testing.assert_close(out_y[1:], out[1:], rtol=0, atol=0)

    def test_global_rms_does_couple_samples(self):
        """Guard rail: shows the old global normaliser is NOT sample-independent.

        If this ever stops holding, the per-sample argument has become a no-op
        and the tests above would pass vacuously.
        """
        torch.manual_seed(3)
        x = torch.randn(4, 7, 5)
        out = rms_normalize(x)
        y = x.clone()
        y[0] *= 100.0
        out_y = rms_normalize(y)
        self.assertGreater((out_y[1:] - out[1:]).abs().max().item(), 1e-3)

    def test_single_sample_matches_global(self):
        """With one sample the per-sample form reduces to the global one exactly."""
        torch.manual_seed(4)
        for shape, dim in (((1, 7, 5), -3), ((1, 6, 6, 5), -4)):
            x = torch.randn(*shape)
            torch.testing.assert_close(
                rms_normalize(x, sample_dim=dim), rms_normalize(x), rtol=0, atol=0
            )


@unittest.skipUnless(
    torch.cuda.is_available(), "DiffusionConditioning uses the fused CUDA LayerNorm"
)
class TestDiffusionConditioningSampleAxis(unittest.TestCase):
    """DiffusionConditioning must accept a per-sample trunk without mixing samples."""

    def test_per_sample_conditioning_matches_shared(self):
        from protenix.model.modules.diffusion import DiffusionConditioning

        torch.manual_seed(5)
        dev = torch.device("cuda")
        n_token, n_sample, c_z, c_s, c_s_in = 6, 3, 8, 12, 9
        mod = (
            DiffusionConditioning(
                c_z=c_z, c_s=c_s, c_s_inputs=c_s_in, c_noise_embedding=16
            )
            .eval()
            .to(dev)
        )

        s_inputs = torch.randn(n_token, c_s_in, device=dev)
        s_trunk = torch.randn(n_token, c_s, device=dev)
        z_trunk = torch.randn(n_token, n_token, c_z, device=dev)
        t_hat = torch.rand(n_sample, device=dev) + 0.5
        relp = _relp_feature(mod, n_token).to(dev)

        with torch.no_grad():
            s_shared, z_shared = mod(
                t_hat, relp, s_inputs, s_trunk, z_trunk, pair_z=None
            )
            # Same trunk, replicated across samples: must give the same answer.
            s_rep = s_trunk.unsqueeze(-3).expand(n_sample, n_token, c_s).contiguous()
            z_rep = (
                z_trunk.unsqueeze(-4)
                .expand(n_sample, n_token, n_token, c_z)
                .contiguous()
            )
            s_per, z_per = mod(t_hat, relp, s_inputs, s_rep, z_rep, pair_z=None)

        self.assertEqual(s_shared.shape, (n_sample, n_token, c_s))
        self.assertEqual(s_per.shape, (n_sample, n_token, c_s))
        torch.testing.assert_close(s_per, s_shared, rtol=1e-5, atol=1e-5)
        self.assertEqual(z_per.shape, (n_sample, n_token, n_token, c_z))
        for i in range(n_sample):
            torch.testing.assert_close(z_per[i], z_shared, rtol=1e-5, atol=1e-5)

    def test_per_sample_conditioning_keeps_samples_separate(self):
        """Perturbing sample i's trunk must not move sample j's conditioning."""
        from protenix.model.modules.diffusion import DiffusionConditioning

        torch.manual_seed(6)
        dev = torch.device("cuda")
        n_token, n_sample, c_z, c_s, c_s_in = 5, 3, 8, 12, 9
        mod = (
            DiffusionConditioning(
                c_z=c_z, c_s=c_s, c_s_inputs=c_s_in, c_noise_embedding=16
            )
            .eval()
            .to(dev)
        )

        s_inputs = torch.randn(n_token, c_s_in, device=dev)
        s_trunk = torch.randn(n_sample, n_token, c_s, device=dev)
        z_trunk = torch.randn(n_sample, n_token, n_token, c_z, device=dev)
        t_hat = torch.rand(n_sample, device=dev) + 0.5
        relp = _relp_feature(mod, n_token).to(dev)

        with torch.no_grad():
            s_a, z_a = mod(t_hat, relp, s_inputs, s_trunk, z_trunk, pair_z=None)
            bumped_s = s_trunk.clone()
            bumped_s[0] += 5.0
            bumped_z = z_trunk.clone()
            bumped_z[0] += 5.0
            s_b, z_b = mod(t_hat, relp, s_inputs, bumped_s, bumped_z, pair_z=None)

        torch.testing.assert_close(s_b[1:], s_a[1:], rtol=0, atol=0)
        torch.testing.assert_close(z_b[1:], z_a[1:], rtol=0, atol=0)
        self.assertGreater((s_b[0] - s_a[0]).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()


class TestChunkSizeCapAndBackoff(unittest.TestCase):
    """Chunk selection must survive a mixed-size batch of targets in one process."""

    @staticmethod
    def _configs(chunk, n_sample=5, epitope="B:9"):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_name="protenix-v2",
            epitope_residue=epitope,
            skip_amp=SimpleNamespace(confidence_head=False, sample_diffusion=True),
            infer_setting=SimpleNamespace(sample_diffusion_chunk_size=chunk),
            sample_diffusion={"N_sample": n_sample},
        )

    def test_ceiling_shrinks_with_target_size(self):
        from runner.inference import update_inference_configs

        cfg = self._configs(5)
        for n_token, expected in ((147, 5), (349, 5), (395, 3), (490, 3), (655, 1)):
            update_inference_configs(cfg, n_token)
            self.assertEqual(
                cfg.infer_setting.sample_diffusion_chunk_size,
                expected,
                f"N_token={n_token}",
            )

    def test_large_target_does_not_pin_later_small_targets(self):
        """The regression this guards: configs are mutated in place per target."""
        from runner.inference import update_inference_configs

        cfg = self._configs(5)
        update_inference_configs(cfg, 900)  # big target -> chunk 1
        self.assertEqual(cfg.infer_setting.sample_diffusion_chunk_size, 1)
        update_inference_configs(cfg, 150)  # small target must go back up
        self.assertEqual(cfg.infer_setting.sample_diffusion_chunk_size, 5)

    def test_explicit_request_is_an_upper_bound(self):
        from runner.inference import update_inference_configs

        cfg = self._configs(2)
        update_inference_configs(cfg, 147)  # ceiling is 5, but 2 was requested
        self.assertEqual(cfg.infer_setting.sample_diffusion_chunk_size, 2)

    def test_no_capping_without_epitope_steering(self):
        """Unguided runs keep upstream Protenix behaviour untouched."""
        from runner.inference import update_inference_configs

        cfg = self._configs(5, epitope=None)
        update_inference_configs(cfg, 2000)
        self.assertEqual(cfg.infer_setting.sample_diffusion_chunk_size, 5)

    @staticmethod
    def _fake_model(blocks_per_ckpt=1):
        """Stand-in exposing the per-module blocks_per_ckpt the restore walks."""
        import torch.nn as nn

        m = nn.Module()
        m.blocks_per_ckpt = blocks_per_ckpt
        parent = nn.Module()
        parent.child = m
        parent.blocks_per_ckpt = blocks_per_ckpt
        return parent

    def test_oom_backoff_halves_until_it_fits(self):
        from runner.inference import _predict_with_oom_backoff

        cfg = self._configs(5)
        seen = []
        outer = self

        class FakeRunner:
            def __init__(self):
                self.calls = 0
                self.model = outer._fake_model()

            def predict(self, data):
                self.calls += 1
                seen.append(cfg.infer_setting.sample_diffusion_chunk_size)
                # The real predict() CONSUMES the feature dict; a retry that
                # reuses it dies on a missing key instead of on memory.
                data.pop("profile")
                if self.calls < 3:
                    raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
                return {"ok": True}

            def update_model_configs(self, new_configs):
                pass

        runner = FakeRunner()
        data = {"profile": [1, 2, 3]}
        out = _predict_with_oom_backoff(runner, cfg, data=data, sample_name="fake")
        self.assertEqual(out, {"ok": True})
        self.assertEqual(seen, [5, 2, 1])
        self.assertIn(
            "profile", data, "the caller's feature dict must not be consumed"
        )

    def test_backoff_restores_checkpointing_before_giving_up(self):
        """blocks_per_ckpt=null buys speed with memory; trade it back, don't lose the target."""
        from runner.inference import _predict_with_oom_backoff

        cfg = self._configs(1)
        outer = self

        class FitsOnlyWithCheckpointing:
            def __init__(self):
                self.model = outer._fake_model(blocks_per_ckpt=None)
                self.attempts = 0

            def predict(self, data):
                self.attempts += 1
                if self.model.blocks_per_ckpt is None:
                    raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
                return {"ok": True}

            def update_model_configs(self, new_configs):
                pass

        runner = FitsOnlyWithCheckpointing()
        out = _predict_with_oom_backoff(
            runner, cfg, data={"profile": [1]}, sample_name="fake"
        )
        self.assertEqual(out, {"ok": True})
        self.assertEqual(runner.attempts, 2)
        self.assertEqual(runner.model.blocks_per_ckpt, 1)
        self.assertEqual(runner.model.child.blocks_per_ckpt, 1)

    def test_oom_at_chunk_one_reraises(self):
        from runner.inference import _predict_with_oom_backoff

        cfg = self._configs(1)
        outer = self

        class AlwaysOOM:
            def __init__(self):
                self.model = outer._fake_model()

            def predict(self, data):
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

            def update_model_configs(self, new_configs):
                pass

        with self.assertRaises(torch.OutOfMemoryError):
            _predict_with_oom_backoff(
                AlwaysOOM(), cfg, data={"profile": [1]}, sample_name="fake"
            )
