import math
import unittest

from sensor_pca.data import DataError, read_csv
from tests.util import write_temp


class ReadCsvTests(unittest.TestCase):
    def test_reads_ids_names_rows(self):
        path = write_temp("id,a,b\nr1,1.0,2.0\nr2,3.0,4.0\n")
        ids, names, rows = read_csv(path)
        self.assertEqual(ids, ["r1", "r2"])
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(rows, [[1.0, 2.0], [3.0, 4.0]])

    def test_rejects_duplicate_column_names(self):
        path = write_temp("id,a,a\nr1,1,2\n")
        with self.assertRaises(DataError):
            read_csv(path)

    def test_rejects_duplicate_ids(self):
        path = write_temp("id,a\nr1,1\nr1,2\n")
        with self.assertRaises(DataError):
            read_csv(path)

    def test_rejects_non_finite_values(self):
        for bad in ("nan", "inf", "-inf"):
            path = write_temp(f"id,a\nr1,{bad}\n")
            with self.assertRaises(DataError, msg=bad):
                read_csv(path)

    def test_rejects_non_numeric_values(self):
        path = write_temp("id,a\nr1,abc\n")
        with self.assertRaises(DataError):
            read_csv(path)

    def test_rejects_ragged_rows(self):
        path = write_temp("id,a,b\nr1,1\n")
        with self.assertRaises(DataError):
            read_csv(path)

    def test_values_are_finite_floats(self):
        path = write_temp("id,a\nr1,1e3\n")
        _, _, rows = read_csv(path)
        self.assertTrue(math.isfinite(rows[0][0]))
        self.assertEqual(rows[0][0], 1000.0)


if __name__ == "__main__":
    unittest.main()
