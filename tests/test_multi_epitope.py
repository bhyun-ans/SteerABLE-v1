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

"""Several epitope sets off one trunk.

`--epitope_residue "B:1,B:2;B:40,B:41"` must become one steered branch per set,
all sampled from the same pairformer output.  These tests pin the pieces that
can be checked without a model: the set parser, the branch naming / resolution,
the per-set enrichment on a toy contact map, and the trunk fingerprint that the
inference loop uses to refuse a branch that would start from a modified trunk.
"""

import unittest

import torch

from protenix.model.gating import (
    BRANCH_STEERED,
    BRANCH_RAW,
    compute_epitope_enrichment,
    is_steered_branch,
    resolve_branches,
    steered_branch_names,
    trunk_fingerprint,
)
from protenix.model.steering import parse_epitope_residue, parse_epitope_sets


class TestParseEpitopeSets(unittest.TestCase):
    def test_single_set_is_one_element(self):
        self.assertEqual(parse_epitope_sets("B:14,B:15"), ["B:14,B:15"])

    def test_sets_split_on_semicolon_in_order(self):
        self.assertEqual(
            parse_epitope_sets("B:14,B:15; B:47,B:48 ;C:3"),
            ["B:14,B:15", "B:47,B:48", "C:3"],
        )

    def test_trailing_and_empty_sets_are_dropped(self):
        self.assertEqual(parse_epitope_sets("B:14;;B:47;"), ["B:14", "B:47"])

    def test_no_set_raises(self):
        with self.assertRaises(ValueError):
            parse_epitope_sets(" ; ")

    def test_each_set_parses_as_before(self):
        sets = parse_epitope_sets("B:14,B:15;C:3")
        self.assertEqual(parse_epitope_residue(sets[0]), [(1, 14), (1, 15)])
        self.assertEqual(parse_epitope_residue(sets[1]), [(2, 3)])

    def test_single_set_parser_rejects_multi_set_string(self):
        with self.assertRaises(ValueError):
            parse_epitope_residue("B:14;B:47")


class TestBranchNaming(unittest.TestCase):
    def test_one_set_uses_the_plain_name(self):
        self.assertEqual(steered_branch_names(1), [BRANCH_STEERED])

    def test_several_sets_are_indexed_in_order(self):
        self.assertEqual(steered_branch_names(3), ["steerable_0", "steerable_1", "steerable_2"])

    def test_zero_sets_raise(self):
        with self.assertRaises(ValueError):
            steered_branch_names(0)

    def test_is_steered_branch(self):
        for name in ("steerable", "steerable_0", "steerable_7"):
            self.assertTrue(is_steered_branch(name))
        self.assertFalse(is_steered_branch(BRANCH_RAW))


class TestResolveBranchesMultiSet(unittest.TestCase):
    def test_single_set_is_unchanged(self):
        self.assertEqual(
            resolve_branches("both", True, n_epitope_sets=1),
            ([BRANCH_STEERED, BRANCH_RAW], None),
        )
        self.assertEqual(
            resolve_branches("steerable", True, n_epitope_sets=1), ([BRANCH_STEERED], None)
        )
        # default argument == one set
        self.assertEqual(resolve_branches("both", True), ([BRANCH_STEERED, BRANCH_RAW], None))

    def test_both_runs_every_set_then_raw_last(self):
        names, routed = resolve_branches("both", True, n_epitope_sets=3)
        self.assertEqual(names, ["steerable_0", "steerable_1", "steerable_2", BRANCH_RAW])
        self.assertIsNone(routed)
        # raw must be last: it is the only branch allowed in-place ops on the trunk
        self.assertEqual(names[-1], BRANCH_RAW)

    def test_steerable_runs_every_set_only(self):
        names, _ = resolve_branches("steerable", True, n_epitope_sets=2)
        self.assertEqual(names, ["steerable_0", "steerable_1"])

    def test_raw_ignores_sets(self):
        self.assertEqual(resolve_branches("raw", True, n_epitope_sets=4), ([BRANCH_RAW], None))

    def test_no_epitope_ignores_sets(self):
        self.assertEqual(
            resolve_branches("both", False, n_epitope_sets=4), ([BRANCH_RAW], None)
        )

    def test_route_refuses_several_sets(self):
        with self.assertRaises(ValueError):
            resolve_branches("route", True, threshold=1.0, enrichment=2.0, n_epitope_sets=2)

    def test_route_still_works_for_one_set(self):
        self.assertEqual(
            resolve_branches("route", True, threshold=1.0, enrichment=2.0, n_epitope_sets=1),
            ([BRANCH_RAW], BRANCH_RAW),
        )


class TestPerSetEnrichment(unittest.TestCase):
    """Two sets scored on one contact map must give two independent readouts."""

    def setUp(self):
        # Chain A (tokens 0-1) is the antibody; chain B (tokens 2-7) the antigen.
        self.asym_id = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
        self.residue_index = torch.tensor([1, 2, 1, 2, 3, 4, 5, 6])
        cp = torch.full((8, 8), 0.01)
        cp[0, 2] = 0.9  # Ab token 0 strongly contacts antigen residue B:1
        cp[1, 3] = 0.8  # Ab token 1 strongly contacts antigen residue B:2
        self.cp = cp

    def test_sets_are_scored_independently(self):
        set_hot = [(1, 1), (1, 2)]
        set_cold = [(1, 5), (1, 6)]
        hot, arr_hot = compute_epitope_enrichment(
            self.cp, self.asym_id, self.residue_index, set_hot, {0}
        )
        cold, arr_cold = compute_epitope_enrichment(
            self.cp, self.asym_id, self.residue_index, set_cold, {0}
        )
        self.assertGreater(hot["enrichment"], 1.0)
        self.assertLess(cold["enrichment"], 1.0)
        # Trunk-level arrays are identical across sets; only the epitope mask differs.
        torch.testing.assert_close(arr_hot["contact_probs"], arr_cold["contact_probs"])
        torch.testing.assert_close(
            arr_hot["epitope_contact_score"], arr_cold["epitope_contact_score"]
        )
        self.assertTrue(torch.equal(arr_hot["ab_token_mask"], arr_cold["ab_token_mask"]))
        self.assertFalse(
            torch.equal(arr_hot["epitope_token_mask"], arr_cold["epitope_token_mask"])
        )
        stacked = torch.stack(
            [arr_hot["epitope_token_mask"], arr_cold["epitope_token_mask"]], dim=0
        )
        self.assertEqual(tuple(stacked.shape), (2, 8))
        self.assertEqual(stacked[0].nonzero().flatten().tolist(), [2, 3])
        self.assertEqual(stacked[1].nonzero().flatten().tolist(), [6, 7])


class TestTrunkFingerprint(unittest.TestCase):
    """The check that makes 'every branch starts from the same trunk' enforceable."""

    def setUp(self):
        torch.manual_seed(0)
        self.s = torch.randn(6, 16)
        self.z = torch.randn(6, 6, 8)

    def test_same_tensors_same_fingerprint(self):
        self.assertEqual(trunk_fingerprint(self.s, self.z), trunk_fingerprint(self.s, self.z))

    def test_out_of_place_update_leaves_fingerprint(self):
        """What the steered branch does: detach + out-of-place add on a copy."""
        fp = trunk_fingerprint(self.s, self.z)
        s_tau = self.s.detach().expand(3, *self.s.shape).contiguous()
        s_tau = s_tau + 0.1 * torch.randn_like(s_tau)
        self.assertGreater((s_tau[0] - self.s).abs().max().item(), 0.0)
        self.assertEqual(trunk_fingerprint(self.s, self.z), fp)

    def test_inplace_write_changes_fingerprint(self):
        """What the raw branch's confidence head does with in-place ops enabled."""
        fp = trunk_fingerprint(self.s, self.z)
        z_alias = self.z.detach()  # shares storage, like ConfidenceHead.forward
        z_alias *= 0
        self.assertNotEqual(trunk_fingerprint(self.s, self.z), fp)

    def test_dtype_of_reduction_is_fp32(self):
        z16 = self.z.to(torch.bfloat16)
        fp = trunk_fingerprint(z16)
        self.assertEqual(len(fp), 2)
        self.assertTrue(all(isinstance(v, float) for v in fp))


if __name__ == "__main__":
    unittest.main()
