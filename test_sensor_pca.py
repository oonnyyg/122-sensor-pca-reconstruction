import csv
import json
import math
import os
import tempfile
import unittest

import sensor_pca as sp


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


class LoadCsvTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "data.csv")

    def tearDown(self):
        self.dir.cleanup()

    def test_duplicate_columns_rejected(self):
        write_csv(self.path, ["id", "a", "a"], [["r1", "1", "2"], ["r2", "3", "4"]])
        with self.assertRaises(sp.PCAError):
            sp.load_csv(self.path)

    def test_non_finite_rejected(self):
        for bad in ("nan", "inf", "-inf", "abc"):
            write_csv(self.path, ["id", "a"], [["r1", bad], ["r2", "1"]])
            with self.assertRaises(sp.PCAError, msg=bad):
                sp.load_csv(self.path)

    def test_basic_load(self):
        write_csv(self.path, ["id", "a", "b"], [["r1", "1.5", "2"], ["r2", "3", "4"]])
        ids, cols, rows = sp.load_csv(self.path)
        self.assertEqual(ids, ["r1", "r2"])
        self.assertEqual(cols, ["a", "b"])
        self.assertEqual(rows, [[1.5, 2.0], [3.0, 4.0]])


class FitTests(unittest.TestCase):
    def test_requires_two_rows(self):
        with self.assertRaises(sp.PCAError):
            sp.fit([[1.0, 2.0]], ["a", "b"], 0.9)

    def test_all_constant_rejected(self):
        with self.assertRaises(sp.PCAError):
            sp.fit([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], ["a", "b"], 0.9)

    def test_bad_ratio_rejected(self):
        rows = [[1.0, 2.0], [2.0, 1.0]]
        for ratio in (0.0, -0.5, 1.5):
            with self.assertRaises(sp.PCAError):
                sp.fit(rows, ["a", "b"], ratio)

    def test_correlated_columns(self):
        # y = 2x exactly: one component must explain ~all variance.
        rows = [[float(i), 2.0 * i] for i in range(10)]
        model = sp.fit(rows, ["x", "y"], 0.999)
        self.assertEqual(len(model["components"]), 1)
        vec = model["components"][0]["vector"]
        self.assertAlmostEqual(abs(vec[0]), 1.0 / math.sqrt(2), places=12)
        self.assertAlmostEqual(abs(vec[1]), 1.0 / math.sqrt(2), places=12)
        self.assertGreater(vec[0], 0.0)  # sign convention: pivot positive
        self.assertAlmostEqual(model["components"][0]["explained_ratio"], 1.0, places=12)

    def test_orthogonal_unit_vectors_sorted(self):
        rows = [[float(i), float(i * i % 7), float((i * 3) % 5)] for i in range(20)]
        model = sp.fit(rows, ["a", "b", "c"], 1.0)
        values = [c["eigenvalue"] for c in model["components"]]
        self.assertEqual(values, sorted(values, reverse=True))
        vecs = [c["vector"] for c in model["components"]]
        for v in vecs:
            self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=10)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dot = sum(a * b for a, b in zip(vecs[i], vecs[j]))
                self.assertAlmostEqual(dot, 0.0, places=10)

    def test_constant_column_excluded_but_kept(self):
        rows = [[float(i), 7.5] for i in range(8)]
        model = sp.fit(rows, ["x", "const"], 0.99)
        self.assertEqual(model["active_columns"], ["x"])
        self.assertEqual(model["means"][1], 7.5)
        self.assertEqual(model["scales"][1], 0.0)
        self.assertEqual(len(model["components"]), 1)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.model_path = os.path.join(self.dir.name, "model.json")
        rows = [[float(i), 2.0 * i, 5.0] for i in range(12)]
        self.columns = ["x", "y", "const"]
        self.model = sp.fit(rows, self.columns, 0.999)
        sp.save_model(self.model, self.model_path)

    def test_model_roundtrip_and_column_reorder(self):
        loaded = sp.load_model(self.model_path)
        self.assertEqual(loaded, self.model)
        # Apply with shuffled column order.
        cols = ["const", "x", "y"]
        rows = [[5.0, 3.0, 6.0], [5.0, 4.0, 8.0]]
        results = sp.apply_model(loaded, ["r1", "r2"], cols, rows)
        # Data lies on the fitted line, so reconstruction is near-exact.
        for res, row in zip(results, [[3.0, 6.0, 5.0], [4.0, 8.0, 5.0]]):
            for got, want in zip(res["reconstruction"], row):
                self.assertAlmostEqual(got, want, places=9)
            self.assertAlmostEqual(res["error"], 0.0, places=9)

    def test_missing_and_extra_columns_rejected(self):
        loaded = sp.load_model(self.model_path)
        with self.assertRaises(sp.PCAError):
            sp.apply_model(loaded, ["r"], ["x", "y"], [[1.0, 2.0]])
        with self.assertRaises(sp.PCAError):
            sp.apply_model(loaded, ["r"], ["x", "y", "const", "z"], [[1.0, 2.0, 5.0, 9.0]])

    def test_constant_column_change_counts_in_error(self):
        loaded = sp.load_model(self.model_path)
        rows = [[3.0, 6.0, 8.0]]  # const moved from 5.0 to 8.0
        res = sp.apply_model(loaded, ["r"], self.columns, rows)[0]
        self.assertAlmostEqual(res["error"], 3.0, places=9)
        self.assertAlmostEqual(res["reconstruction"][2], 5.0, places=12)

    def test_off_line_error(self):
        loaded = sp.load_model(self.model_path)
        res = sp.apply_model(loaded, ["r"], self.columns, [[3.0, 7.0, 5.0]])[0]
        # Point off the y=2x line: residual in standardized space, rescaled.
        self.assertGreater(res["error"], 0.0)
        self.assertAlmostEqual(
            res["error"],
            math.sqrt(sum((a - r) ** 2 for a, r in zip([3.0, 7.0, 5.0], res["reconstruction"]))),
            places=12,
        )


class CliTests(unittest.TestCase):
    def test_fit_then_apply(self):
        with tempfile.TemporaryDirectory() as d:
            train = os.path.join(d, "train.csv")
            new = os.path.join(d, "new.csv")
            model = os.path.join(d, "model.json")
            out = os.path.join(d, "out.csv")
            write_csv(train, ["id", "x", "y"], [[f"t{i}", str(i), str(2 * i)] for i in range(10)])
            write_csv(new, ["id", "y", "x"], [["n1", "6.0", "3.0"]])
            self.assertEqual(sp.main(["fit", train, model, "--variance-ratio", "0.99"]), 0)
            self.assertTrue(os.path.exists(model))
            self.assertEqual(sp.main(["apply", new, model, out]), 0)
            with open(out, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0][0], "id")
            self.assertIn("score_pc1", rows[0])
            self.assertIn("reconstruction_error", rows[0])
            self.assertEqual(rows[1][0], "n1")
            self.assertAlmostEqual(float(rows[1][-1]), 0.0, places=9)
            with open(model, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["format"], "sensor-pca-model/1")


if __name__ == "__main__":
    unittest.main()
