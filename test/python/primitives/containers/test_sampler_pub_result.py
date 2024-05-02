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


"""Unit tests for SamplerPubResult."""

from test import QiskitTestCase

import numpy as np

from qiskit.primitives.containers import BitArray, DataBin, SamplerPubResult


class SamplerPubResultCase(QiskitTestCase):
    """Test the SamplerPubResult class."""

    def test_construction(self):
        """Test that the constructor works."""
        ba = BitArray.from_samples(["00", "11"], 2)
        counts = {"00": 1, "11": 1}
        data_bin = DataBin(a=ba, b=ba)
        pub_result = SamplerPubResult(data_bin)
        self.assertEqual(pub_result.data.a.get_counts(), counts)
        self.assertEqual(pub_result.data.b.get_counts(), counts)
        self.assertEqual(pub_result.metadata, {})

        pub_result = SamplerPubResult(data_bin, {"x": 1})
        self.assertEqual(pub_result.data.a.get_counts(), counts)
        self.assertEqual(pub_result.data.b.get_counts(), counts)
        self.assertEqual(pub_result.metadata, {"x": 1})

    def test_repr(self):
        """Test that the repr doesn't fail"""
        # we are primarily interested in making sure some future change doesn't cause the repr to
        # raise an error. it is more sensible for humans to detect a deficiency in the formatting
        # itself, should one be uncovered
        ba = BitArray.from_samples(["00", "11"], 2)
        data_bin = DataBin(a=ba, b=ba)
        self.assertTrue(repr(SamplerPubResult(data_bin)).startswith("SamplerPubResult"))
        self.assertTrue(repr(SamplerPubResult(data_bin, {"x": 1})).startswith("SamplerPubResult"))