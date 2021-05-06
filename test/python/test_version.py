# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/version.py"""

from qiskit import __qiskit_version__
from qiskit import __version__
from qiskit.test import QiskitTestCase


class TestVersion(QiskitTestCase):
    """Tests for qiskit/version.py"""

    def test_qiskit_version(self):
        """Test qiskit-version sets the correct version for terra."""
        self.assertEqual(__version__, __qiskit_version__["qiskit-terra"])
