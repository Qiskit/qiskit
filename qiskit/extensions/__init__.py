# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================================
Quantum Circuit Extensions (:mod:`qiskit.extensions`)
=====================================================

.. currentmodule:: qiskit.extensions

Unitary Extensions
==================

.. autosummary::
   :toctree: ../stubs/

   SingleQubitUnitary

Simulator Extensions
====================

.. autosummary::
   :toctree: ../stubs/

   Snapshot

Exceptions
==========

The additional gates in this module will tend to raise a custom exception when they encounter
problems.

.. autoexception:: ExtensionError
"""

import warnings

# import all standard gates
from qiskit.circuit.library.standard_gates import *
from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.circuit.barrier import Barrier

from .exceptions import ExtensionError
from .quantum_initializer import (
    Initialize,
    SingleQubitUnitary,
    UCPauliRotGate,
    UCRXGate,
    UCRYGate,
    UCRZGate,
)
from .simulator import Snapshot


warnings.warn(
    "The qiskit.extensions module is pending deprecation since Qiskit 0.45.0. It will be deprecated "
    "in a following release, no sooner than 3 months after the 0.45.0 release.",
    stacklevel=2,
    category=PendingDeprecationWarning,
)
