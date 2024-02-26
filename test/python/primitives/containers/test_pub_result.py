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


"""Unit tests for PubResult."""

from qiskit.primitives.containers import PubResult, make_data_bin
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class PubResultCase(QiskitTestCase):
    """Test the PubResult class."""

    def test_construction(self):
        """Test that the constructor works."""
        data_bin = make_data_bin((("a", float), ("b", int)))
        pub_result = PubResult(data_bin(a=1.0, b=2))
        self.assertEqual(pub_result.data.a, 1.0)
        self.assertEqual(pub_result.data.b, 2)
        self.assertEqual(pub_result.metadata, {})

        pub_result = PubResult(data_bin(a=1.0, b=2), {"x": 1})
        self.assertEqual(pub_result.data.a, 1.0)
        self.assertEqual(pub_result.data.b, 2)
        self.assertEqual(pub_result.metadata, {"x": 1})

    def test_repr(self):
        """Test that the repr doesn't fail"""
        # we are primarily interested in making sure some future change doesn't cause the repr to
        # raise an error. it is more sensible for humans to detect a deficiency in the formatting
        # itself, should one be uncovered
        data_bin = make_data_bin((("a", float), ("b", int)))
        self.assertTrue(repr(PubResult(data_bin(a=1.0, b=2))).startswith("PubResult"))
        self.assertTrue(repr(PubResult(data_bin(a=1.0, b=2), {"x": 1})).startswith("PubResult"))
