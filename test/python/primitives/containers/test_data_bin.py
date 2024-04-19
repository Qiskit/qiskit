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

from qiskit.primitives.containers import make_data_bin
from qiskit.primitives.containers.data_bin import DataBin, DataBinMeta
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class DataBinTestCase(QiskitTestCase):
    """Test the DataBin class."""

    def test_make_databin(self):
        """Test the make_databin() function."""
        data_bin_cls = make_data_bin(
            [("alpha", npt.NDArray[np.uint16]), ("beta", np.ndarray)], shape=(10, 20)
        )

        self.assertTrue(issubclass(type(data_bin_cls), DataBinMeta))
        self.assertTrue(issubclass(data_bin_cls, DataBin))

        alpha = np.empty((10, 20), dtype=np.uint16)
        beta = np.empty((10, 20), dtype=int)
        my_bin = data_bin_cls(alpha=alpha, beta=beta)
        self.assertEqual(len(my_bin), 2)
        self.assertTrue(np.all(my_bin.alpha == alpha))
        self.assertTrue(np.all(my_bin.beta == beta))
        self.assertTrue("alpha=" in str(my_bin))
        self.assertTrue(str(my_bin).startswith("DataBin"))
        self.assertEqual(my_bin._FIELDS, ("alpha", "beta"))
        self.assertEqual(my_bin._FIELD_TYPES, (np.ndarray, np.ndarray))

        my_bin = data_bin_cls(beta=beta, alpha=alpha)
        self.assertTrue(np.all(my_bin.alpha == alpha))
        self.assertTrue(np.all(my_bin.beta == beta))

    def test_make_databin_no_shape(self):
        """Test the make_databin() function with no shape."""
        data_bin_cls = make_data_bin([("alpha", dict), ("beta", int)])

        self.assertTrue(issubclass(type(data_bin_cls), DataBinMeta))
        self.assertTrue(issubclass(data_bin_cls, DataBin))

        my_bin = data_bin_cls(alpha={1: 2}, beta=5)
        self.assertEqual(my_bin.alpha, {1: 2})
        self.assertEqual(my_bin.beta, 5)
        self.assertTrue("alpha=" in str(my_bin))
        self.assertTrue(">" not in str(my_bin))
        self.assertEqual(my_bin._FIELDS, ("alpha", "beta"))
        self.assertEqual(my_bin._FIELD_TYPES, (dict, int))

    def test_make_databin_no_fields(self):
        """Test the make_data_bin() function when no fields are given."""
        data_bin_cls = make_data_bin([])
        data_bin = data_bin_cls()
        self.assertEqual(len(data_bin), 0)
