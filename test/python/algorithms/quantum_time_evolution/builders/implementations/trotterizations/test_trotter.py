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
""" Test Trotter. """
import unittest

import numpy as np

from qiskit.algorithms.quantum_time_evolution.builders.implementations.pauli_trotter_evolution_op_builder import (
    PauliTrotterEvolutionOpBuilder,
)
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.suzuki import (
    Suzuki,
)
from qiskit.circuit import ParameterVector
from test.python.opflow import QiskitOpflowTestCase
from qiskit.opflow import (
    X,
    Z,
    I,
    Y,
    CX,
    Zero,
    H,
)


class TestSuzuki(QiskitOpflowTestCase):
    """Trotter tests."""


if __name__ == "__main__":
    unittest.main()
