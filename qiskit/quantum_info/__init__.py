# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
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
   Clifford
   ScalarOp
   SparsePauliOp
   CNOTDihedral
   PauliList
   PauliTable
   StabilizerTable
   pauli_basis
   pauli_group

States
======

.. autosummary::
   :toctree: ../stubs/

   Statevector
   DensityMatrix
   StabilizerState

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
   random_pauli
   random_clifford
   random_quantum_channel
   random_cnotdihedral
   random_pauli_table
   random_stabilizer_table

Analysis
=========

.. autosummary::
   :toctree: ../stubs/

   hellinger_distance
   hellinger_fidelity

Synthesis
=========

.. autosummary::
   :toctree: ../stubs/

   OneQubitEulerDecomposer
   TwoQubitBasisDecomposer
   two_qubit_cnot_decompose
   Quaternion
   decompose_clifford
"""

from .operators import Operator, ScalarOp, Pauli, Clifford, SparsePauliOp
from .operators import PauliList, PauliTable, StabilizerTable, pauli_basis, pauli_group
from .operators.channel import Choi, SuperOp, Kraus, Stinespring, Chi, PTM
from .operators.measures import process_fidelity, average_gate_fidelity, gate_error, diamond_norm
from .operators.dihedral import CNOTDihedral

from .states import Statevector, DensityMatrix, StabilizerState
from .states import (
    partial_trace,
    state_fidelity,
    purity,
    entropy,
    concurrence,
    entanglement_of_formation,
    mutual_information,
    shannon_entropy,
)

from .random import (
    random_quantum_channel,
    random_unitary,
    random_clifford,
    random_pauli,
    random_pauli_table,
    random_stabilizer_table,
    random_hermitian,
    random_statevector,
    random_density_matrix,
    random_cnotdihedral,
)

from .synthesis import (
    OneQubitEulerDecomposer,
    TwoQubitBasisDecomposer,
    two_qubit_cnot_decompose,
    Quaternion,
    decompose_clifford,
)

from .analysis import hellinger_distance, hellinger_fidelity
