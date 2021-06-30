# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Operators."""

from .operator import Operator
from .scalar_op import ScalarOp
from .channel import Choi, SuperOp, Kraus, Stinespring, Chi, PTM
from .measures import process_fidelity, average_gate_fidelity, gate_error, diamond_norm
from .symplectic import Clifford, Pauli, PauliList, SparsePauliOp, PauliTable, StabilizerTable
from .symplectic import pauli_basis
from .pauli import pauli_group
from .dihedral import CNOTDihedral
