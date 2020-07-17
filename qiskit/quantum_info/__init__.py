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
   Clifford
   ScalarOp
   SparsePauliOp
   Pauli
   pauli_group
   Quaternion
   PauliTable
   StabilizerTable
   pauli_basis

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

   average_gate_fidelity
   process_fidelity
   gate_error
   diamond_norm
   state_fidelity
   purity
   concurrence
   entropy
   entanglement_of_formation
   mutual_information

Utility Functions
=================

.. autosummary::
   :toctree: ../stubs/

   partial_trace
   shannon_entropy

Random
======

.. autosummary::
   :toctree: ../stubs/

   random_statevector
   random_density_matrix
   random_unitary
   random_hermitian
   random_clifford
   random_quantum_channel
   random_pauli_table
   random_stabilizer_table
   random_state

Analysis
=========

.. autosummary::
   :toctree: ../stubs/

   hellinger_fidelity

Synthesis
=========

.. autosummary::
   :toctree: ../stubs/

   OneQubitEulerDecomposer
   TwoQubitBasisDecomposer
   two_qubit_cnot_decompose
   euler_angles_1q
"""

from .operators import Operator, ScalarOp
from .operators.pauli import Pauli, pauli_group
from .operators.quaternion import Quaternion
from .operators.channel import Choi, SuperOp, Kraus, Stinespring, Chi, PTM
from .operators.measures import (process_fidelity,
                                 average_gate_fidelity,
                                 gate_error,
                                 diamond_norm)
from .operators.symplectic import (Clifford, SparsePauliOp,
                                   PauliTable, StabilizerTable)
from .operators.symplectic import pauli_basis

from .states import Statevector, DensityMatrix
from .states import (partial_trace, state_fidelity, purity, entropy,
                     concurrence, entanglement_of_formation,
                     mutual_information, shannon_entropy)

from .random import (random_quantum_channel, random_unitary,
                     random_clifford, random_pauli_table,
                     random_stabilizer_table,
                     random_hermitian, random_statevector,
                     random_density_matrix, random_state)

from .synthesis import (OneQubitEulerDecomposer, TwoQubitBasisDecomposer,
                        two_qubit_cnot_decompose, euler_angles_1q)

from .analysis import hellinger_fidelity
