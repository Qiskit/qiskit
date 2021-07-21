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

"""PauliOp Class """

from math import pi
from typing import Dict, List, Optional, Set, Union, cast

import numpy as np
from scipy.sparse import spmatrix

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.library import IGate, RXGate, RYGate, RZGate, XGate, YGate, ZGate
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector


class PauliOp(PrimitiveOp):
    """Class for Operators backed by Terra's ``Pauli`` module."""

    primitive: Pauli

    def __init__(self, primitive: Pauli, coeff: Union[complex, ParameterExpression] = 1.0) -> None:
        """
        Args:
            primitive: The Pauli which defines the behavior of the underlying function.
            coeff: A coefficient multiplying the primitive.

        Raises:
            TypeError: invalid parameters.
        """
        if not isinstance(primitive, Pauli):
            raise TypeError(f"PauliOp can only be instantiated with Paulis, not {type(primitive)}")
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {"Pauli"}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive)

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over operators with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        if isinstance(other, PauliOp) and self.primitive == other.primitive:
            return PauliOp(self.primitive, coeff=self.coeff + other.coeff)

        # pylint: disable=cyclic-import
        from .pauli_sum_op import PauliSumOp

        if (
            isinstance(other, PauliOp)
            and isinstance(self.coeff, (int, float, complex))
            and isinstance(other.coeff, (int, float, complex))
        ):
            return PauliSumOp(
                SparsePauliOp(self.primitive, coeffs=[self.coeff])
                + SparsePauliOp(other.primitive, coeffs=[other.coeff])
            )

        if isinstance(other, PauliSumOp) and isinstance(self.coeff, (int, float, complex)):
            return PauliSumOp(SparsePauliOp(self.primitive, coeffs=[self.coeff])) + other

        return SummedOp([self, other])

    def adjoint(self) -> "PauliOp":
        return PauliOp(self.primitive, coeff=self.coeff.conjugate())

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, PauliOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def _expand_dim(self, num_qubits: int) -> "PauliOp":
        return PauliOp(Pauli("I" * num_qubits).expand(self.primitive), coeff=self.coeff)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        # Both Paulis
        if isinstance(other, PauliOp):
            return PauliOp(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)

        # pylint: disable=cyclic-import
        from .pauli_sum_op import PauliSumOp

        if isinstance(other, PauliSumOp):
            new_primitive = SparsePauliOp(self.primitive).tensor(other.primitive)
            return PauliSumOp(new_primitive, coeff=self.coeff * other.coeff)

        from .circuit_op import CircuitOp

        if isinstance(other, CircuitOp):
            return self.to_circuit_op().tensor(other)

        return TensoredOp([self, other])

    def permute(self, permutation: List[int]) -> "PauliOp":
        """Permutes the sequence of Pauli matrices.

        Args:
            permutation: A list defining where each Pauli should be permuted. The Pauli at index
                j of the primitive should be permuted to position permutation[j].

        Returns:
              A new PauliOp representing the permuted operator. For operator (X ^ Y ^ Z) and
              indices=[1,2,4], it returns (X ^ I ^ Y ^ Z ^ I).

        Raises:
            OpflowError: if indices do not define a new index for each qubit.
        """
        pauli_string = self.primitive.__str__()
        length = max(permutation) + 1  # size of list must be +1 larger then its max index
        new_pauli_list = ["I"] * length
        if len(permutation) != self.num_qubits:
            raise OpflowError(
                "List of indices to permute must " "have the same size as Pauli Operator"
            )
        for i, index in enumerate(permutation):
            new_pauli_list[-index - 1] = pauli_string[-i - 1]
        return PauliOp(Pauli("".join(new_pauli_list)), self.coeff)

    def compose(
        self, other: OperatorBase, permutation: Optional[List[int]] = None, front: bool = False
    ) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(PauliOp, new_self)

        if front:
            return other.compose(new_self)
        # If self is identity, just return other.
        if not any(new_self.primitive.x + new_self.primitive.z):
            return other * new_self.coeff

        # Both Paulis
        if isinstance(other, PauliOp):
            product = new_self.primitive.dot(other.primitive)
            return PrimitiveOp(product, coeff=new_self.coeff * other.coeff)

        # pylint: disable=cyclic-import
        from .pauli_sum_op import PauliSumOp

        if isinstance(other, PauliSumOp):
            return PauliSumOp(
                SparsePauliOp(new_self.primitive).dot(other.primitive),
                coeff=new_self.coeff * other.coeff,
            )

        # pylint: disable=cyclic-import
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from .circuit_op import CircuitOp

        if isinstance(other, (CircuitOp, CircuitStateFn)):
            return new_self.to_circuit_op().compose(other)

        return super(PauliOp, new_self).compose(other)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", True, self.num_qubits, massive)
        return self.primitive.to_matrix() * self.coeff

    def to_spmatrix(self) -> spmatrix:
        """Returns SciPy sparse matrix representation of the Operator.

        Returns:
            CSR sparse matrix representation of the Operator.

        Raises:
            ValueError: invalid parameters.
        """
        return self.primitive.to_matrix(sparse=True) * self.coeff

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return f"{self.coeff} * {prim_str}"

    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        if front is None:
            return self.to_matrix_op()

        # pylint: disable=cyclic-import
        from ..list_ops.list_op import ListOp
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.state_fn import StateFn
        from .circuit_op import CircuitOp

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            new_front = front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        else:

            if self.num_qubits != front.num_qubits:
                raise ValueError(
                    "eval does not support operands with differing numbers of qubits, "
                    "{} and {}, respectively.".format(self.num_qubits, front.num_qubits)
                )

            if isinstance(front, DictStateFn):

                new_dict: Dict[str, complex] = {}
                corrected_x_bits = self.primitive.x[::-1]
                corrected_z_bits = self.primitive.z[::-1]

                for bstr, v in front.primitive.items():
                    bitstr = np.fromiter(bstr, dtype=int).astype(bool)
                    new_b_str = np.logical_xor(bitstr, corrected_x_bits)
                    new_str = "".join(map(str, 1 * new_b_str))
                    z_factor = np.product(1 - 2 * np.logical_and(bitstr, corrected_z_bits))
                    y_factor = np.product(
                        np.sqrt(1 - 2 * np.logical_and(corrected_x_bits, corrected_z_bits) + 0j)
                    )
                    new_dict[new_str] = (v * z_factor * y_factor) + new_dict.get(new_str, 0)
                    # The coefficient consists of:
                    #   1. the coefficient of *this* PauliOp (self)
                    #   2. the coefficient of the evaluated DictStateFn (front)
                    #   3. AND acquires the phase of the internal primitive. This is necessary to
                    #      ensure that (X @ Z) and (-iY) return the same result.
                    new_front = StateFn(
                        new_dict, coeff=self.coeff * front.coeff * (-1j) ** self.primitive.phase
                    )

            elif isinstance(front, StateFn) and front.is_measurement:
                raise ValueError("Operator composed with a measurement is undefined.")

            # Composable types with PauliOp
            elif isinstance(front, (PauliOp, CircuitOp, CircuitStateFn)):
                new_front = self.compose(front)

            # Covers VectorStateFn and OperatorStateFn
            elif isinstance(front, StateFn):
                new_front = self.to_matrix_op().eval(front.to_matrix_op())

        return new_front

    def exp_i(self) -> OperatorBase:
        """Return a ``CircuitOp`` equivalent to e^-iH for this operator H."""
        # if only one qubit is significant, we can perform the evolution
        corrected_x = self.primitive.x[::-1]
        corrected_z = self.primitive.z[::-1]
        sig_qubits = np.logical_or(corrected_x, corrected_z)
        if np.sum(sig_qubits) == 0:
            # e^I is just a global phase, but we can keep track of it! Should we?
            # For now, just return identity
            return PauliOp(self.primitive)
        if np.sum(sig_qubits) == 1:
            sig_qubit_index = sig_qubits.tolist().index(True)
            coeff = (
                np.real(self.coeff)
                if not isinstance(self.coeff, ParameterExpression)
                else self.coeff
            )

            from .circuit_op import CircuitOp

            # Y rotation
            if corrected_x[sig_qubit_index] and corrected_z[sig_qubit_index]:
                rot_op = CircuitOp(RYGate(2 * coeff))
            # Z rotation
            elif corrected_z[sig_qubit_index]:
                rot_op = CircuitOp(RZGate(2 * coeff))
            # X rotation
            elif corrected_x[sig_qubit_index]:
                rot_op = CircuitOp(RXGate(2 * coeff))

            # pylint: disable=cyclic-import
            from ..operator_globals import I

            left_pad = I.tensorpower(sig_qubit_index)
            right_pad = I.tensorpower(self.num_qubits - sig_qubit_index - 1)
            # Need to use overloaded operators here in case left_pad == I^0
            return left_pad ^ rot_op ^ right_pad
        else:
            from ..evolutions.evolved_op import EvolvedOp

            return EvolvedOp(self)

    def to_circuit(self) -> QuantumCircuit:

        pauli = self.primitive.to_label()[-self.num_qubits :]
        phase = self.primitive.phase

        qc = QuantumCircuit(self.num_qubits)
        if pauli == "I" * self.num_qubits:
            qc.global_phase = -phase * pi / 2
            return qc

        if self.num_qubits == 1:
            gate = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}[pauli]
        else:
            gate = PauliGate(pauli)
        qc.append(gate, range(self.num_qubits))

        if not phase:
            return qc

        qc.global_phase = -phase * pi / 2
        return qc

    def to_instruction(self) -> Instruction:
        # TODO should we just do the following because performance of adding and deleting IGates
        #  doesn't matter?
        # (Reduce removes extra IGates).
        # return PrimitiveOp(self.primitive.to_instruction(), coeff=self.coeff).reduce()

        return self.primitive.to_instruction()

    def to_pauli_op(self, massive: bool = False) -> "PauliOp":
        return self
