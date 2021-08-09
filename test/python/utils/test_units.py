# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for unit conversion functions."""

from qiskit.test import QiskitTestCase
from qiskit.utils import apply_prefix, detach_prefix


class TestUnitConversion(QiskitTestCase):
    """Test the unit conversion utilities."""

    def test_apply_prefix(self):
        """Test applying prefix to value."""
        ref_values = [
            ([1.0, "THz"], 1e12),
            ([1.0, "GHz"], 1e9),
            ([1.0, "MHz"], 1e6),
            ([1.0, "kHz"], 1e3),
            ([1.0, "mHz"], 1e-3),
            ([1.0, "µHz"], 1e-6),
            ([1.0, "uHz"], 1e-6),
            ([1.0, "nHz"], 1e-9),
            ([1.0, "pHz"], 1e-12),
        ]

        for args, ref_ret in ref_values:
            self.assertEqual(apply_prefix(*args), ref_ret)

    def test_not_convert_meter(self):
        """Test not apply prefix to meter."""
        self.assertEqual(apply_prefix(1.0, "m"), 1.0)

    def test_detach_prefix(self):
        """Test detach prefix from the value."""
        ref_values = [
            (1e12, (1.0, "T")),
            (1e11, (100.0, "G")),
            (1e10, (10.0, "G")),
            (1e9, (1.0, "G")),
            (1e8, (100.0, "M")),
            (1e7, (10.0, "M")),
            (1e6, (1.0, "M")),
            (1e5, (100.0, "k")),
            (1e4, (10.0, "k")),
            (1e3, (1.0, "k")),
            (100, (100.0, "")),
            (10, (10.0, "")),
            (1.0, (1.0, "")),
            (0.1, (100.0, "m")),
            (0.01, (10.0, "m")),
            (1e-3, (1.0, "m")),
            (1e-4, (100.0, "μ")),
            (1e-5, (10.0, "μ")),
            (1e-6, (1.0, "μ")),
            (1e-7, (100.0, "n")),
            (1e-8, (10.0, "n")),
            (1e-9, (1.0, "n")),
            (1e-10, (100.0, "p")),
            (1e-11, (10.0, "p")),
            (1e-12, (1.0, "p")),
        ]

        for arg, ref_rets in ref_values:
            self.assertTupleEqual(detach_prefix(arg), ref_rets)
