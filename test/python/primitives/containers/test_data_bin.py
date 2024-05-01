# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Unit tests for DataBin."""


import numpy as np
import numpy.typing as npt

from qiskit.primitives.containers.data_bin import DataBin
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class DataBinTestCase(QiskitTestCase):
    """Test the DataBin class."""

    def test_data_bin(self):
        """Test DataBin function basic access."""
        alpha = np.empty((10, 20), dtype=np.uint16)
        beta = np.empty((10, 20), dtype=int)
        my_bin = DataBin(alpha=alpha, beta=beta)
        self.assertEqual(len(my_bin), 2)
        self.assertTrue(np.all(my_bin.alpha == alpha))
        self.assertTrue(np.all(my_bin.beta == beta))
        self.assertTrue("alpha=" in str(my_bin))
        self.assertTrue(str(my_bin).startswith("DataBin"))
        self.assertEqual(my_bin._FIELDS, ("alpha", "beta"))
        self.assertEqual(my_bin._FIELD_TYPES, (np.ndarray, np.ndarray))

        my_bin = DataBin(beta=beta, alpha=alpha)
        self.assertTrue(np.all(my_bin.alpha == alpha))
        self.assertTrue(np.all(my_bin.beta == beta))

    def test_make_databin_no_shape(self):
        """Test DataBin with no shape."""
        my_bin = DataBin(alpha={1: 2}, beta=5)
        self.assertEqual(my_bin.alpha, {1: 2})
        self.assertEqual(my_bin.beta, 5)
        self.assertTrue("alpha=" in str(my_bin))
        self.assertTrue(">" not in str(my_bin))
        self.assertEqual(my_bin._FIELDS, ("alpha", "beta"))
        self.assertEqual(my_bin._FIELD_TYPES, (dict, int))

    def test_make_databin_no_fields(self):
        """Test DataBin when no fields are given."""
        data_bin = DataBin()
        self.assertEqual(len(data_bin), 0)

    def test_make_databin_mapping(self):
        """Test the make_data_bin() function with mapping features."""
        data_bin = DataBin(alpha=10, beta={1: 2})
        self.assertEqual(len(data_bin), 2)

        with self.subTest("iterator"):
            iterator = iter(data_bin)
            key = next(iterator)
            self.assertEqual(key, "alpha")
            key = next(iterator)
            self.assertEqual(key, "beta")
            with self.assertRaises(StopIteration):
                _ = next(iterator)

        with self.subTest("keys"):
            lst = data_bin.keys()
            key = lst[0]
            self.assertEqual(key, "alpha")
            key = lst[1]
            self.assertEqual(key, "beta")

        with self.subTest("values"):
            lst = data_bin.values()
            val = lst[0]
            self.assertIsInstance(val, int)
            self.assertEqual(val, 10)
            val = lst[1]
            self.assertIsInstance(val, dict)
            self.assertEqual(val, {1: 2})

        with self.subTest("items"):
            lst = data_bin.items()
            key, val = lst[0]
            self.assertEqual(key, "alpha")
            self.assertIsInstance(val, int)
            self.assertEqual(val, 10)
            key, val = lst[1]
            self.assertEqual(key, "beta")
            self.assertIsInstance(val, dict)
            self.assertEqual(val, {1: 2})

        with self.subTest("contains"):
            self.assertIn("alpha", data_bin)
            self.assertIn("beta", data_bin)
            self.assertNotIn("gamma", data_bin)

        with self.subTest("getitem"):
            val = data_bin["alpha"]
            self.assertIsInstance(val, int)
            self.assertEqual(val, 10)
            val = data_bin["beta"]
            self.assertIsInstance(val, dict)
            self.assertEqual(val, {1: 2})

        with self.subTest("error"):
            with self.assertRaises(KeyError):
                _ = data_bin["gamma"]
