# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Zero Operator Class """

from typing import Dict, List, Optional, Set, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.quantum_info import Statevector
from qiskit.utils.validation import validate_min


class ZeroOp(OperatorBase):
    def __init__(self, num_qubits: int):
        validate_min("num_qubits", num_qubits, 1)
        self._num_qubits = num_qubits
        super().__init__()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def primitive_strings(self) -> Set[str]:
        return {"Zero"}

    def reduce(self):
        return self

    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, "OperatorBase", Statevector]
        ] = None,
    ) -> complex:
        return 0

    def to_matrix(self):
        return np.zeros((num_qubits, num_qubits))

    def to_matrix_op(self):

        return MatrixOp(self.to_matrix, 0)

    def to_circuit_op(self):

        return CircuitOp(QuantumCircuit(self.num_qubits), 0)

    def add(self, other):
        return other

    def adjoint(self):
        return self

    def __eq__(self, other):
        if other == 0:
            return True
        if not isinstance(other, OperatorBase):
            return NotImplemented
        return self.equals(cast(OperatorBase, other))

    def equals(self, other: "OperatorBase") -> bool:
        if not isinstance(other, OperatorBase):
            return NotImplemented

        if isinstance(other, PauliSumOp) and other.is_zero():
            return True

        if isinstance(self, (ListOp, PrimitiveOp, StateFn)):
            return self.reduce().coeff == 0
        return False

    def mul(self, other):
        return self

    def tensor(self, other):
        return ZeroOp(self.num_qubits + other.num_qubits)

    def tensorpower(self, other):
        return ZeroOp(self.num_qubits * other)

    @property
    def parameters(self):
        return {}

    def assign_parameters(self, param_dict):
        return self

    def _expand_dim(self, num_qubits: int) -> OperatorBase:
        return ZeroOp(self.num_qubits + num_qubits)

    def permute(self, permutation: List[int]) -> "OperatorBase":
        return self

    def compose(self, other):
        return self

    def __repr__(self):
        return f"ZeroOp({self.num_qubits})"

    def __str__(self):
        return "0"
