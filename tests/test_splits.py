import argparse
import copy
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from benchmark import (
    evaluate_fuzzy_scores, prepare_splits, run_single_seed_experiment,
    select_split, split_fraction,
)
from transe import TransE
from utils import RealKG


def options(**overrides):
    values = dict(
        train_fraction=1.0, val_fraction=1.0, test_fraction=1.0,
        train_sample_size=0, eval_sample_size=0,
        val_sample_size=None, test_sample_size=None, eval_batch_size=2,
        embedding_dim=4, margin=2.0, p_norm=1, alpha=0.1,
        lambda_fuzzy=0.1, fuzzy_bias=0.0, lr=0.005, epochs=1, batch_size=2,
        models=["standard", "sigmoid", "exponential", "gaussian"],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.kg = SimpleNamespace(
            train_triples=[(i, 0, i + 1) for i in range(100)],
            val_triples=[(i, 1, i + 1) for i in range(20)],
            test_triples=[(i, 2, i + 1) for i in range(30)],
        )

    def test_defaults_use_complete_independent_copies(self):
        before = copy.deepcopy(self.kg)
        selected = prepare_splits(self.kg, options(), 42)
        for actual, original in zip(selected, vars(self.kg).values()):
            self.assertEqual(actual, original)
            self.assertIsNot(actual, original)
        selected[0].reverse()
        self.assertEqual(vars(self.kg), vars(before))

    def test_fraction_cap_precedence_and_split_boundaries(self):
        selected = prepare_splits(self.kg, options(
            train_fraction=0.5, train_sample_size=40, eval_sample_size=5,
            val_sample_size=0, test_fraction=0.5, test_sample_size=12,
        ), 42)
        self.assertEqual([len(x) for x in selected], [40, 20, 12])
        for actual, original in zip(selected, vars(self.kg).values()):
            self.assertTrue(set(actual) <= set(original))
            self.assertEqual(len(actual), len(set(actual)))
        self.assertFalse(set(selected[0]) & set(selected[1]))
        self.assertFalse(set(selected[0]) & set(selected[2]))

    def test_sampling_is_repeatable_and_does_not_consume_global_rng(self):
        state = random.getstate()
        first = select_split(self.kg.train_triples, 0.5, seed=42)
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, select_split(self.kg.train_triples, 0.5, seed=42))
        self.assertNotEqual(first, select_split(self.kg.train_triples, 0.5, seed=100))

    def test_invalid_or_small_selections(self):
        for fraction in (0, -1, 1.1, float('nan')):
            with self.assertRaises(argparse.ArgumentTypeError):
                split_fraction(fraction)
        with self.assertRaises(ValueError):
            select_split([])
        self.assertEqual(len(select_split(self.kg.train_triples, 0.001)), 1)
        self.assertEqual(select_split(self.kg.val_triples, cap=-1), self.kg.val_triples)

    def test_batched_membership_preserves_scores(self):
        model = TransE(101, 3, embedding_dim=4)
        triples = self.kg.val_triples
        full = evaluate_fuzzy_scores(model, triples, triples, 'cpu', 100)
        batched = evaluate_fuzzy_scores(model, triples, triples, 'cpu', 3)
        for expected, actual in zip(full, batched):
            np.testing.assert_allclose(expected, actual, rtol=1e-6)

    def test_real_training_preserves_splits_and_repeats(self):
        # Real model training and all metrics, without a dataset download.
        kg = RealKG.__new__(RealKG)
        kg.train_triples = [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 1, 4)]
        kg.val_triples = [(0, 1, 2), (1, 1, 3)]
        kg.test_triples = [(2, 0, 4), (4, 1, 0)]
        before = copy.deepcopy(vars(kg))
        known = set(kg.train_triples + kg.val_triples + kg.test_triples)
        runs = [run_single_seed_experiment(
            'toy', 'medium', 42, options(), torch.device('cpu'), kg, known, 5, 2
        ) for _ in range(2)]
        self.assertEqual(vars(kg), before)
        self.assertEqual(runs[0][1], runs[1][1])
        for model in runs[0][0]:
            for metric, value in runs[0][0][model].items():
                if metric != 'Train Time (s)':
                    self.assertTrue(np.isfinite(value))
                    self.assertAlmostEqual(value, runs[1][0][model][metric])


if __name__ == '__main__':
    unittest.main()
