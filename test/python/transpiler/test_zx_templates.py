# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the zx_templates."""

import numpy as np

from qiskit.quantum_info import Operator
from qiskit.circuit.library.templates import zx_zz1, zx_zz2, zx_zy, zx_cy
from qiskit.test import QiskitTestCase


class TestZXTemplates(QiskitTestCase):
    """Test the parametric templates."""

    def test_templates(self):
        """Test that the templates compose to the identity."""

        self.assertTrue(np.allclose(Operator(zx_zy(0.456)).data, np.eye(4)))

        data = Operator(zx_cy(0.456)).data
        self.assertTrue(np.allclose(data, data[0, 0] * np.eye(4)))

        data = Operator(zx_zz1(0.456)).data
        self.assertTrue(np.allclose(data, data[0, 0]*np.eye(4)))

        data = Operator(zx_zz2(0.456)).data
        self.assertTrue(np.allclose(data, data[0, 0]*np.eye(4)))
