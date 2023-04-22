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

"""CircuitStateFn Class"""


from typing import Dict, List, Optional, Set, Union, cast

import numpy as np

from qiskit import BasicAer, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import IGate, StatePreparation
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func


class CircuitStateFn(StateFn):
    r"""
    Deprecated: A class for state functions and measurements which are defined by the action of a
    QuantumCircuit starting from \|0⟩, and stored using Terra's ``QuantumCircuit`` class.
    """
    primitive: QuantumCircuit

    # TODO allow normalization somehow?
    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: Union[QuantumCircuit, Instruction] = None,
        coeff: Union[complex, ParameterExpression] = 1.0,
        is_measurement: bool = False,
        from_operator: bool = False,
    ) -> None:
        """
        Args:
            primitive: The ``QuantumCircuit`` (or ``Instruction``, which will be converted) which
                defines the behavior of the underlying function.
            coeff: A coefficient multiplying the state function.
            is_measurement: Whether the StateFn is a measurement operator.
            from_operator: if True the StateFn is derived from OperatorStateFn. (Default: False)

        Raises:
            TypeError: Unsupported primitive, or primitive has ClassicalRegisters.
        """
        if isinstance(primitive, Instruction):
            qc = QuantumCircuit(primitive.num_qubits)
            qc.append(primitive, qargs=range(primitive.num_qubits))
            primitive = qc

        if not isinstance(primitive, QuantumCircuit):
            raise TypeError(
                "CircuitStateFn can only be instantiated "
                "with QuantumCircuit, not {}".format(type(primitive))
            )

        if len(primitive.clbits) != 0:
            raise TypeError("CircuitOp does not support QuantumCircuits with ClassicalRegisters.")

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

        self.from_operator = from_operator

    @staticmethod
    def from_dict(density_dict: dict) -> "CircuitStateFn":
        """Construct the CircuitStateFn from a dict mapping strings to probability densities.

        Args:
            density_dict: The dict representing the desired state.

        Returns:
            The CircuitStateFn created from the dict.
        """
        # If the dict is sparse (elements <= qubits), don't go
        # building a statevector to pass to Qiskit's
        # initializer, just create a sum.
        if len(density_dict) <= len(list(density_dict.keys())[0]):
            statefn_circuits = []
            for bstr, prob in density_dict.items():
                qc = QuantumCircuit(len(bstr))
                # NOTE: Reversing endianness!!
                for (index, bit) in enumerate(reversed(bstr)):
                    if bit == "1":
                        qc.x(index)
                sf_circuit = CircuitStateFn(qc, coeff=prob)
                statefn_circuits += [sf_circuit]
            if len(statefn_circuits) == 1:
                return statefn_circuits[0]
            else:
                return cast(CircuitStateFn, SummedOp(cast(List[OperatorBase], statefn_circuits)))
        else:
            sf_dict = StateFn(density_dict)
            return CircuitStateFn.from_vector(sf_dict.to_matrix())

    @staticmethod
    def from_vector(statevector: np.ndarray) -> "CircuitStateFn":
        """Construct the CircuitStateFn from a vector representing the statevector.

        Args:
            statevector: The statevector representing the desired state.

        Returns:
            The CircuitStateFn created from the vector.
        """
        normalization_coeff = np.linalg.norm(statevector)
        normalized_sv = statevector / normalization_coeff
        return CircuitStateFn(StatePreparation(normalized_sv), coeff=normalization_coeff)

    def primitive_strings(self) -> Set[str]:
        return {"QuantumCircuit"}

    @property
    def settings(self) -> Dict:
        """Return settings."""
        data = super().settings
        data["from_operator"] = self.from_operator
        return data

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over operators with different numbers of qubits, "
                "{} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        if isinstance(other, CircuitStateFn) and self.primitive == other.primitive:
            return CircuitStateFn(self.primitive, coeff=self.coeff + other.coeff)

        # Covers all else.
        return SummedOp([self, other])

    def adjoint(self) -> "CircuitStateFn":
        try:
            inverse = self.primitive.inverse()
        except CircuitError as missing_inverse:
            raise OpflowError(
                "Failed to take the inverse of the underlying circuit, the circuit "
                "is likely not unitary and can therefore not be inverted."
            ) from missing_inverse

        return CircuitStateFn(
            inverse, coeff=self.coeff.conjugate(), is_measurement=(not self.is_measurement)
        )

    def compose(
        self, other: OperatorBase, permutation: Optional[List[int]] = None, front: bool = False
    ) -> OperatorBase:
        if not self.is_measurement and not front:
            raise ValueError(
                "Composition with a Statefunctions in the first operand is not defined."
            )
        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self.from_operator = self.from_operator

        if front:
            return other.compose(new_self)

        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            op_circuit_self = CircuitOp(self.primitive)

            # Avoid reimplementing compose logic
            composed_op_circs = cast(CircuitOp, op_circuit_self.compose(other.to_circuit_op()))

            # Returning CircuitStateFn
            return CircuitStateFn(
                composed_op_circs.primitive,
                is_measurement=self.is_measurement,
                coeff=self.coeff * other.coeff,
                from_operator=self.from_operator,
            )

        if isinstance(other, CircuitStateFn) and self.is_measurement:
            # pylint: disable=cyclic-import
            from ..operator_globals import Zero

            return self.compose(CircuitOp(other.primitive)).compose(
                (Zero ^ self.num_qubits) * other.coeff
            )

        return ComposedOp([new_self, other])

    def tensor(self, other: OperatorBase) -> Union["CircuitStateFn", TensoredOp]:
        r"""
        Return tensor product between self and other, overloaded by ``^``.
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, Plus.tensor(Zero)
        produces a \|+⟩ on qubit 0 and a \|0⟩ on qubit 1, or \|+⟩⨂\|0⟩, but would produce
        a QuantumCircuit like:

            \|0⟩--
            \|+⟩--

        Because Terra prints circuits and results with qubit 0 at the end of the string or circuit.

        Args:
            other: The ``OperatorBase`` to tensor product with self.

        Returns:
            An ``OperatorBase`` equivalent to the tensor product of self and other.
        """
        if isinstance(other, CircuitStateFn) and other.is_measurement == self.is_measurement:
            # Avoid reimplementing tensor, just use CircuitOp's
            c_op_self = CircuitOp(self.primitive, self.coeff)
            c_op_other = CircuitOp(other.primitive, other.coeff)
            c_op = c_op_self.tensor(c_op_other)
            if isinstance(c_op, CircuitOp):
                return CircuitStateFn(
                    primitive=c_op.primitive,
                    coeff=c_op.coeff,
                    is_measurement=self.is_measurement,
                )
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """
        Return numpy matrix of density operator, warn if more than 16 qubits to
        force the user to set
        massive=True if they want such a large matrix. Generally big methods like this
        should require the use of a
        converter, but in this case a convenience method for quick hacking and access
        to classical tools is
        appropriate.
        """
        OperatorBase._check_massive("to_density_matrix", True, self.num_qubits, massive)
        # Rely on VectorStateFn's logic here.
        return VectorStateFn(self.to_matrix(massive=massive) * self.coeff).to_density_matrix()

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", False, self.num_qubits, massive)

        # Need to adjoint to get forward statevector and then reverse
        if self.is_measurement:
            return np.conj(self.adjoint().to_matrix(massive=massive))
        qc = self.to_circuit(meas=False)
        statevector_backend = BasicAer.get_backend("statevector_simulator")
        transpiled = transpile(qc, statevector_backend, optimization_level=0)
        statevector = statevector_backend.run(transpiled).result().get_statevector()
        from ..operator_globals import EVAL_SIG_DIGITS

        return np.round(statevector * self.coeff, decimals=EVAL_SIG_DIGITS)

    def __str__(self) -> str:
        qc = cast(CircuitStateFn, self.reduce()).to_circuit()
        prim_str = str(qc.draw(output="text"))
        if self.coeff == 1.0:
            return "{}(\n{}\n)".format(
                "CircuitStateFn" if not self.is_measurement else "CircuitMeasurement", prim_str
            )
        else:
            return "{}(\n{}\n) * {}".format(
                "CircuitStateFn" if not self.is_measurement else "CircuitMeasurement",
                prim_str,
                self.coeff,
            )

    def assign_parameters(self, param_dict: dict) -> Union["CircuitStateFn", ListOp]:
        param_value = self.coeff
        qc = self.primitive
        if isinstance(self.coeff, ParameterExpression) or self.primitive.parameters:
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if isinstance(self.coeff, ParameterExpression) and self.coeff.parameters <= set(
                unrolled_dict.keys()
            ):
                param_instersection = set(unrolled_dict.keys()) & self.coeff.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                param_value = float(self.coeff.bind(binds))
            # & is set intersection, check if any parameters in unrolled are present in circuit
            # This is different from bind_parameters in Terra because they check for set equality
            if set(unrolled_dict.keys()) & self.primitive.parameters:
                # Only bind the params found in the circuit
                param_instersection = set(unrolled_dict.keys()) & self.primitive.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                qc = self.to_circuit().assign_parameters(binds)
        return self.__class__(qc, coeff=param_value, is_measurement=self.is_measurement)

    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        if front is None:
            vector_state_fn = self.to_matrix_op().eval()
            return vector_state_fn

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                "Cannot compute overlap with StateFn or Operator if not Measurement. Try taking "
                "sf.adjoint() first to convert to measurement."
            )

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        # Composable with circuit
        if isinstance(front, (PauliOp, CircuitOp, MatrixOp, CircuitStateFn)):
            new_front = self.compose(front)
            return new_front.eval()

        return self.to_matrix_op().eval(front)

    def to_circuit(self, meas: bool = False) -> QuantumCircuit:
        """Return QuantumCircuit representing StateFn"""
        if meas:
            meas_qc = self.primitive.copy()
            meas_qc.add_register(ClassicalRegister(self.num_qubits))
            meas_qc.measure(qubit=range(self.num_qubits), cbit=range(self.num_qubits))
            return meas_qc
        else:
            return self.primitive

    def to_circuit_op(self) -> OperatorBase:
        """Return ``StateFnCircuit`` corresponding to this StateFn."""
        return self

    def to_instruction(self):
        """Return Instruction corresponding to primitive."""
        return self.primitive.to_instruction()

    # TODO specify backend?
    def sample(
        self, shots: int = 1024, massive: bool = False, reverse_endianness: bool = False
    ) -> dict:
        """
        Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability.
        """
        OperatorBase._check_massive("sample", False, self.num_qubits, massive)
        qc = self.to_circuit(meas=True)
        qasm_backend = BasicAer.get_backend("qasm_simulator")
        transpiled = transpile(qc, qasm_backend, optimization_level=0)
        counts = qasm_backend.run(transpiled, shots=shots).result().get_counts()
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))

    # Warning - modifying primitive!!
    def reduce(self) -> "CircuitStateFn":
        if self.primitive.data is not None:
            # Need to do this from the end because we're deleting items!
            for i in reversed(range(len(self.primitive.data))):
                gate = self.primitive.data[i].operation
                # Check if Identity or empty instruction (need to check that type is exactly
                # Instruction because some gates have lazy gate.definition population)
                # pylint: disable=unidiomatic-typecheck
                if isinstance(gate, IGate) or (
                    type(gate) == Instruction and gate.definition.data == []
                ):
                    del self.primitive.data[i]
        return self

    def _expand_dim(self, num_qubits: int) -> "CircuitStateFn":
        # this is equivalent to self.tensor(identity_operator), but optimized for better performance
        # just like in tensor method, qiskit endianness is reversed here
        return self.permute(list(range(num_qubits, num_qubits + self.num_qubits)))

    def permute(self, permutation: List[int]) -> "CircuitStateFn":
        r"""
        Permute the qubits of the circuit.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j of the circuit should be permuted to position permutation[j].

        Returns:
            A new CircuitStateFn containing the permuted circuit.
        """
        new_qc = QuantumCircuit(max(permutation) + 1).compose(self.primitive, qubits=permutation)
        return CircuitStateFn(new_qc, coeff=self.coeff, is_measurement=self.is_measurement)
