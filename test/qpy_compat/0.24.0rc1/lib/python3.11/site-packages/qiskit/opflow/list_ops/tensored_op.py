# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TensoredOp Class"""

from functools import partial, reduce
from typing import List, Union, cast, Dict

import numpy as np

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func


class TensoredOp(ListOp):
    """Deprecated: A class for lazily representing tensor products of Operators. Often Operators
    cannot be efficiently tensored to one another, but may be manipulated further so that they can be
    later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be tensored together, and therefore if they reach a point in which they can be, such as after
    conversion to QuantumCircuits, they can be reduced by tensor product."""

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        oplist: List[OperatorBase],
        coeff: Union[complex, ParameterExpression] = 1.0,
        abelian: bool = False,
    ) -> None:
        """
        Args:
            oplist: The Operators being tensored.
            coeff: A coefficient multiplying the operator
            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.
        """
        super().__init__(oplist, combo_fn=partial(reduce, np.kron), coeff=coeff, abelian=abelian)

    @property
    def num_qubits(self) -> int:
        return sum(op.num_qubits for op in self.oplist)

    @property
    def distributive(self) -> bool:
        return False

    @property
    def settings(self) -> Dict:
        """Return settings."""
        return {"oplist": self._oplist, "coeff": self._coeff, "abelian": self._abelian}

    def _expand_dim(self, num_qubits: int) -> "TensoredOp":
        """Appends I ^ num_qubits to ``oplist``. Choice of PauliOp as
        identity is arbitrary and can be substituted for other PrimitiveOp identity.

        Returns:
            TensoredOp expanded with identity operator.
        """
        # pylint: disable=cyclic-import
        from ..operator_globals import I

        return TensoredOp(self.oplist + [I ^ num_qubits], coeff=self.coeff)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, TensoredOp):
            return TensoredOp(self.oplist + other.oplist, coeff=self.coeff * other.coeff)
        return TensoredOp(self.oplist + [other], coeff=self.coeff)

    # TODO eval should partial trace the input into smaller StateFns each of size
    #  op.num_qubits for each op in oplist. Right now just works through matmul.
    def eval(
        self, front: Union[str, dict, np.ndarray, OperatorBase, Statevector] = None
    ) -> Union[OperatorBase, complex]:
        if self._is_empty():
            return 0.0
        return cast(Union[OperatorBase, complex], self.to_matrix_op().eval(front=front))

    # Try collapsing list or trees of tensor products.
    # TODO do this smarter
    def reduce(self) -> OperatorBase:
        reduced_ops = [op.reduce() for op in self.oplist]
        if self._is_empty():
            return self.__class__([], coeff=self.coeff, abelian=self.abelian)
        reduced_ops = reduce(lambda x, y: x.tensor(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, ListOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)

    def to_circuit(self) -> QuantumCircuit:
        """Returns the quantum circuit, representing the tensored operator.

        Returns:
            The circuit representation of the tensored operator.

        Raises:
            OpflowError: for operators where a single underlying circuit can not be produced.
        """
        circuit_op = self.to_circuit_op()
        # pylint: disable=cyclic-import
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..primitive_ops.primitive_op import PrimitiveOp

        if isinstance(circuit_op, (PrimitiveOp, CircuitStateFn)):
            return circuit_op.to_circuit()
        raise OpflowError(
            "Conversion to_circuit supported only for operators, where a single "
            "underlying circuit can be produced."
        )

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", True, self.num_qubits, massive)

        mat = self.coeff * reduce(np.kron, [np.asarray(op.to_matrix()) for op in self.oplist])
        return np.asarray(mat, dtype=complex)
