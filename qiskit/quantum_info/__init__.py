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

.. _quantum_info_operators:

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

.. _quantum_info_states:

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
   commutator
   anti_commutator
   double_commutator

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
   random_pauli_list
   random_stabilizer_table

Analysis
=========

.. autosummary::
   :toctree: ../stubs/

   hellinger_distance
   hellinger_fidelity
   Z2Symmetries

Synthesis
=========

.. autosummary::
   :toctree: ../stubs/

   OneQubitEulerDecomposer
   TwoQubitBasisDecomposer
   two_qubit_cnot_decompose
   Quaternion
   decompose_clifford
   XXDecomposer
"""

from .analysis import hellinger_distance, hellinger_fidelity, Z2Symmetries
from .operators import (
    Clifford,
    Operator,
    Pauli,
    PauliList,
    PauliTable,
    ScalarOp,
    SparsePauliOp,
    StabilizerTable,
    anti_commutator,
    commutator,
    double_commutator,
    pauli_basis,
)
from .operators.channel import PTM, Chi, Choi, Kraus, Stinespring, SuperOp
from .operators.dihedral import CNOTDihedral
from .operators.measures import average_gate_fidelity, diamond_norm, gate_error, process_fidelity
from .random import (
    random_clifford,
    random_cnotdihedral,
    random_density_matrix,
    random_hermitian,
    random_pauli,
    random_pauli_list,
    random_pauli_table,
    random_quantum_channel,
    random_stabilizer_table,
    random_statevector,
    random_unitary,
)
from .states import (
    DensityMatrix,
    StabilizerState,
    Statevector,
    concurrence,
    entanglement_of_formation,
    entropy,
    mutual_information,
    partial_trace,
    purity,
    shannon_entropy,
    state_fidelity,
)
from .synthesis import (
    OneQubitEulerDecomposer,
    Quaternion,
    TwoQubitBasisDecomposer,
    XXDecomposer,
    decompose_clifford,
    two_qubit_cnot_decompose,
)
