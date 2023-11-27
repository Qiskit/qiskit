# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the functions in ``utils.experimental``."""
import warnings

from qiskit.utils import ExperimentalQiskitAPI
from qiskit.test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestExperiemntalWarning(QiskitTestCase):
    """Test plain ExperimentalQiskitAPI warning"""

    def test_experimental_warning(self):
        """Test ExperimentalQiskitAPI warning"""
        with self.assertWarns(ExperimentalQiskitAPI) as cm:
            warnings.warn("qiskit.experimental.api", ExperimentalQiskitAPI)
        self.assertEqual(len(cm.warnings), 1)
        self.assertEqual(
            str(cm.warnings[-1].message),
            "Calling qiskit.experimental.api is experimental "
            "and it might be changed or removed at any point",
        )
