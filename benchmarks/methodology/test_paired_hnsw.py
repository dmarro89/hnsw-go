#!/usr/bin/env python3

import importlib.util
import math
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).with_name("paired_hnsw.py")
spec = importlib.util.spec_from_file_location("paired_hnsw", MODULE_PATH)
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


class MethodologyTests(unittest.TestCase):
    def test_geomean(self):
        self.assertAlmostEqual(paired.geomean([1.0, 4.0]), 2.0)

    def test_classify_win(self):
        self.assertEqual(paired.classify(1.03, 1.01, 1.05, 0.02), "win")

    def test_classify_regression(self):
        self.assertEqual(paired.classify(0.96, 0.94, 0.98, 0.02), "regression")

    def test_small_positive_effect_is_inconclusive(self):
        self.assertEqual(paired.classify(1.015, 1.005, 1.025, 0.02), "inconclusive")

    def test_noisy_large_effect_is_inconclusive(self):
        self.assertEqual(paired.classify(1.04, 0.99, 1.08, 0.02), "inconclusive")

    def test_bootstrap_ci_contains_stable_ratio(self):
        lo, hi = paired.bootstrap_ci([1.03] * 6, samples=500)
        self.assertTrue(math.isclose(lo, 1.03, rel_tol=1e-12))
        self.assertTrue(math.isclose(hi, 1.03, rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
