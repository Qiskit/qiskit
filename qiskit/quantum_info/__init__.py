# -*- coding: utf-8 -*-

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
================================================
Quantum Information (:mod:`qiskit.quantum_info`)
================================================

.. currentmodule:: qiskit.quantum_info

Operators
=========

.. autosummary::
   :toctree: ../stubs/

   Operator
   Pauli
   pauli_group
   Quaternion

States
======

.. autosummary::
   :toctree: ../stubs/

   Statevector
   DensityMatrix

Channels
========

.. autosummary::
   :toctree: ../stubs/

   Choi
   SuperOp
   Kraus
   Stinespring
   Chi
   PTM

Measures
========

.. autosummary::
   :toctree: ../stubs/

   state_fidelity

Random
======

.. autosummary::
   :toctree: ../stubs/

   random_unitary
   random_state
   random_density_matrix

Analysis
=========

.. autosummary::
   :toctree: ../stubs/

   hellinger_fidelity

Synthesis
=========

.. autosummary::
   :toctree: ../stubs/

   euler_angles_1q
   two_qubit_cnot_decompose
   TwoQubitBasisDecomposer

"""

from .operators.operator import Operator
from .operators.pauli import Pauli, pauli_group
from .operators.quaternion import Quaternion
from .operators.channel import Choi, SuperOp, Kraus, Stinespring, Chi, PTM
from .operators.measures import process_fidelity
from .states import Statevector, DensityMatrix
from .states.states import basis_state, projector, purity
from .states.measures import state_fidelity
from .random import random_unitary, random_state, random_density_matrix
from .synthesis import (TwoQubitBasisDecomposer, euler_angles_1q,
                        two_qubit_cnot_decompose)
from .analysis import hellinger_fidelity
