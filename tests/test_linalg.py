import math
import unittest

from sensor_pca.linalg import (
    canonicalize_sign,
    column_means,
    column_stdevs,
    covariance_matrix,
    jacobi_eigh,
    standardize,
)


class StatsTests(unittest.TestCase):
    def test_sample_stdev_uses_n_minus_one(self):
        rows = [[1.0], [2.0], [3.0]]
        means = column_means(rows)
        stdevs = column_stdevs(rows, means)
        self.assertAlmostEqual(stdevs[0], 1.0)  # sqrt(2 / (3 - 1))

    def test_standardize_constant_column_maps_to_zero(self):
        rows = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
        means = column_means(rows)
        stdevs = column_stdevs(rows, means)
        z = standardize(rows, means, stdevs)
        self.assertTrue(all(row[1] == 0.0 for row in z))


class CovarianceTests(unittest.TestCase):
    def test_standardized_covariance_has_unit_diagonal(self):
        rows = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 4.0]]
        means = column_means(rows)
        stdevs = column_stdevs(rows, means)
        cov = covariance_matrix(standardize(rows, means, stdevs))
        self.assertAlmostEqual(cov[0][0], 1.0)
        self.assertAlmostEqual(cov[1][1], 1.0)
        self.assertAlmostEqual(cov[0][1], cov[1][0])


class JacobiTests(unittest.TestCase):
    def test_known_eigenpairs(self):
        values, vectors = jacobi_eigh([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(values[0], 3.0)
        self.assertAlmostEqual(values[1], 1.0)
        inv = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(abs(vectors[0][0]), inv)
        self.assertAlmostEqual(abs(vectors[1][0]), inv)

    def test_eigenvectors_are_orthonormal(self):
        matrix = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 2.0]]
        _, vectors = jacobi_eigh(matrix)
        for a in range(3):
            for b in range(3):
                dot = sum(vectors[i][a] * vectors[i][b] for i in range(3))
                self.assertAlmostEqual(dot, 1.0 if a == b else 0.0, places=10)

    def test_repeated_eigenvalues_are_stable(self):
        values, _ = jacobi_eigh([[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(values, [2.0, 2.0])
        again, _ = jacobi_eigh([[2.0, 0.0], [0.0, 2.0]])
        self.assertEqual(values, again)


class SignTests(unittest.TestCase):
    def test_largest_abs_component_becomes_positive(self):
        self.assertEqual(canonicalize_sign([-0.8, 0.2]), [0.8, -0.2])

    def test_tie_broken_by_column_order(self):
        self.assertEqual(canonicalize_sign([-0.5, 0.5]), [0.5, -0.5])
        self.assertEqual(canonicalize_sign([0.5, -0.5]), [0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
