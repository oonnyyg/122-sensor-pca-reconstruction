import csv
import os
import tempfile
import unittest

from sensor_pca.cli import main
from tests.util import write_temp

TRAIN = (
    "id,x,y,c\n"
    "r1,1.0,2.0,7.0\n"
    "r2,2.0,4.0,7.0\n"
    "r3,3.0,6.0,7.0\n"
    "r4,4.0,8.0,7.0\n"
    "r5,5.0,10.0,7.0\n"
)


class CliTests(unittest.TestCase):
    def test_fit_then_apply_round_trip(self):
        train = write_temp(TRAIN)
        model_path = os.path.join(tempfile.mkdtemp(), "model.json")
        rc = main(
            ["fit", "--input", train, "--model", model_path,
             "--variance-ratio", "0.99"]
        )
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(model_path))

        new_data = write_temp("id,c,y,x\nn1,7.0,4.0,2.0\nn2,9.0,6.0,3.0\n")
        out_dir = tempfile.mkdtemp()
        scores = os.path.join(out_dir, "scores.csv")
        recon = os.path.join(out_dir, "recon.csv")
        rc = main(
            ["apply", "--model", model_path, "--input", new_data,
             "--scores", scores, "--reconstruction", recon]
        )
        self.assertEqual(rc, 0)

        with open(scores, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["id", "pc1"])
        self.assertEqual([r[0] for r in rows[1:]], ["n1", "n2"])

        with open(recon, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["id", "x", "y", "c", "reconstruction_error"])
        # n2 changed the constant column 7 -> 9, so its error must be >= 2.
        self.assertAlmostEqual(float(rows[1][4]), 0.0, places=9)
        self.assertGreaterEqual(float(rows[2][4]), 2.0 - 1e-9)

    def test_fit_rejects_single_row(self):
        path = write_temp("id,x\nr1,1.0\n")
        model_path = os.path.join(tempfile.mkdtemp(), "m.json")
        rc = main(
            ["fit", "--input", path, "--model", model_path,
             "--variance-ratio", "0.9"]
        )
        self.assertEqual(rc, 2)

    def test_apply_rejects_missing_column(self):
        train = write_temp(TRAIN)
        model_path = os.path.join(tempfile.mkdtemp(), "model.json")
        main(["fit", "--input", train, "--model", model_path,
              "--variance-ratio", "0.99"])
        bad = write_temp("id,x,y\nn1,1.0,2.0\n")
        out_dir = tempfile.mkdtemp()
        rc = main(
            ["apply", "--model", model_path, "--input", bad,
             "--scores", os.path.join(out_dir, "s.csv"),
             "--reconstruction", os.path.join(out_dir, "r.csv")]
        )
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
