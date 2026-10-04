import unittest

from sensor_pca.data import DataError
from sensor_pca.model import PCAModel
from tests.util import write_temp

COLUMNS = ["x", "y", "c"]
# x and y are perfectly correlated; c is constant.
ROWS = [
    [1.0, 2.0, 7.0],
    [2.0, 4.0, 7.0],
    [3.0, 6.0, 7.0],
    [4.0, 8.0, 7.0],
    [5.0, 10.0, 7.0],
]


class FitTests(unittest.TestCase):
    def test_correlated_columns_reduce_to_one_component(self):
        model = PCAModel.fit(COLUMNS, ROWS, 0.99)
        self.assertEqual(len(model.components), 1)
        self.assertAlmostEqual(sum(model.explained_ratio), 1.0)
        comp = model.components[0]
        self.assertAlmostEqual(comp[0], comp[1])
        self.assertGreater(comp[0], 0.0)
        self.assertAlmostEqual(comp[2], 0.0)

    def test_constant_column_mean_is_preserved(self):
        model = PCAModel.fit(COLUMNS, ROWS, 0.99)
        self.assertTrue(model.constant[2])
        self.assertEqual(model.means[2], 7.0)

    def test_all_constant_columns_rejected(self):
        with self.assertRaises(DataError):
            PCAModel.fit(["a", "b"], [[1.0, 2.0], [1.0, 2.0]], 0.9)

    def test_variance_ratio_bounds(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(DataError, msg=str(bad)):
                PCAModel.fit(COLUMNS, ROWS, bad)

    def test_minimum_components_meet_target(self):
        rows = [
            [1.0, 1.1, 3.0],
            [2.0, 1.9, 2.0],
            [3.0, 3.2, 5.0],
            [4.0, 3.8, 1.0],
            [5.0, 5.1, 4.0],
        ]
        model = PCAModel.fit(["a", "b", "d"], rows, 0.9)
        cumulative = sum(model.explained_ratio)
        self.assertGreaterEqual(cumulative, 0.9)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.model = PCAModel.fit(COLUMNS, ROWS, 0.99)

    def test_column_order_alignment(self):
        scores_a, _, _ = self.model.apply(COLUMNS, [[2.0, 4.0, 7.0]])
        scores_b, _, _ = self.model.apply(["c", "x", "y"], [[7.0, 2.0, 4.0]])
        self.assertEqual(scores_a, scores_b)

    def test_missing_and_extra_columns_rejected(self):
        with self.assertRaises(DataError):
            self.model.apply(["x", "y"], [[1.0, 2.0]])
        with self.assertRaises(DataError):
            self.model.apply(["x", "y", "c", "z"], [[1.0, 2.0, 7.0, 9.0]])

    def test_constant_column_change_counts_in_error(self):
        _, _, errors_same = self.model.apply(COLUMNS, [[3.0, 6.0, 7.0]])
        _, _, errors_diff = self.model.apply(COLUMNS, [[3.0, 6.0, 9.0]])
        self.assertAlmostEqual(errors_same[0], 0.0, places=9)
        self.assertAlmostEqual(errors_diff[0], 2.0)

    def test_reconstruction_restores_constant_value(self):
        _, recon, _ = self.model.apply(COLUMNS, [[3.0, 6.0, 9.0]])
        self.assertEqual(recon[0][2], 7.0)

    def test_full_rank_reconstruction_is_exact(self):
        model = PCAModel.fit(COLUMNS, ROWS, 1.0)
        _, recon, errors = model.apply(COLUMNS, [[2.5, 5.0, 7.0]])
        self.assertAlmostEqual(errors[0], 0.0, places=9)
        self.assertAlmostEqual(recon[0][0], 2.5, places=9)
        self.assertAlmostEqual(recon[0][1], 5.0, places=9)


class PersistenceTests(unittest.TestCase):
    def test_model_round_trip_through_json(self):
        model = PCAModel.fit(COLUMNS, ROWS, 0.99)
        path = write_temp("", suffix=".json")
        model.save(path)
        loaded = PCAModel.load(path)
        self.assertEqual(loaded.columns, model.columns)
        self.assertEqual(loaded.means, model.means)
        self.assertEqual(loaded.components, model.components)
        scores_a, _, _ = model.apply(COLUMNS, [[2.0, 4.0, 7.0]])
        scores_b, _, _ = loaded.apply(COLUMNS, [[2.0, 4.0, 7.0]])
        self.assertEqual(scores_a, scores_b)

    def test_rejects_foreign_json(self):
        path = write_temp('{"format": "other"}', suffix=".json")
        with self.assertRaises(DataError):
            PCAModel.load(path)


if __name__ == "__main__":
    unittest.main()
