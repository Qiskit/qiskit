# -*- coding: utf-8 -*-

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

""" CircuitStateFn Class """


from typing import Union, Set, List, cast
import numpy as np

from qiskit import QuantumCircuit, BasicAer, execute, ClassicalRegister
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.extensions import Initialize
from qiskit.circuit.library import IGate

from ..operator_base import OperatorBase
from ..list_ops.summed_op import SummedOp
from .state_fn import StateFn


class CircuitStateFn(StateFn):
    r"""
    A class for state functions and measurements which are defined by the action of a
    QuantumCircuit starting from \|0⟩, and stored using Terra's ``QuantumCircuit`` class.
    """

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[QuantumCircuit, Instruction] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The ``QuantumCircuit`` (or ``Instruction``, which will be converted) which
                defines the behavior of the underlying function.
            coeff: A coefficient multiplying the state function.
            is_measurement: Whether the StateFn is a measurement operator.

        Raises:
            TypeError: Unsupported primitive, or primitive has ClassicalRegisters.
        """
        if isinstance(primitive, Instruction):
            qc = QuantumCircuit(primitive.num_qubits)
            qc.append(primitive, qargs=range(primitive.num_qubits))
            primitive = qc

        if not isinstance(primitive, QuantumCircuit):
            raise TypeError('CircuitStateFn can only be instantiated '
                            'with QuantumCircuit, not {}'.format(type(primitive)))

        if len(primitive.clbits) != 0:
            raise TypeError('CircuitOp does not support QuantumCircuits with ClassicalRegisters.')

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    @staticmethod
    def from_dict(density_dict: dict) -> 'CircuitStateFn':
        """ Construct the CircuitStateFn from a dict mapping strings to probability densities.

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
                    if bit == '1':
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
    def from_vector(statevector: np.ndarray) -> 'CircuitStateFn':
        """ Construct the CircuitStateFn from a vector representing the statevector.

        Args:
            statevector: The statevector representing the desired state.

        Returns:
            The CircuitStateFn created from the vector.

        Raises:
            ValueError: If a vector with complex values is passed, which the Initializer cannot
            handle.
        """
        normalization_coeff = np.linalg.norm(statevector)
        normalized_sv = statevector / normalization_coeff
        if not np.all(np.abs(statevector) == statevector):
            # TODO maybe switch to Isometry?
            raise ValueError('Qiskit circuit Initializer cannot handle non-positive statevectors.')
        return CircuitStateFn(Initialize(normalized_sv), coeff=normalization_coeff)

    def primitive_strings(self) -> Set[str]:
        return {'QuantumCircuit'}

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError('Sum over operators with different numbers of qubits, '
                             '{} and {}, is not well '
                             'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, CircuitStateFn) and self.primitive == other.primitive:
            return CircuitStateFn(self.primitive, coeff=self.coeff + other.coeff)

        # Covers all else.
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return CircuitStateFn(self.primitive.inverse(),
                              coeff=np.conj(self.coeff),
                              is_measurement=(not self.is_measurement))

    def compose(self, other: OperatorBase) -> OperatorBase:
        if not self.is_measurement:
            raise ValueError(
                'Composition with a Statefunctions in the first operand is not defined.')

        new_self, other = self._check_zero_for_composition_and_expand(other)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..primitive_ops.circuit_op import CircuitOp
        from ..primitive_ops.pauli_op import PauliOp
        from ..primitive_ops.matrix_op import MatrixOp

        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            op_circuit_self = CircuitOp(self.primitive)

            # Avoid reimplementing compose logic
            composed_op_circs = op_circuit_self.compose(other.to_circuit_op())

            # Returning CircuitStateFn
            return CircuitStateFn(composed_op_circs.primitive,  # type: ignore
                                  is_measurement=self.is_measurement,
                                  coeff=self.coeff * other.coeff)

        if isinstance(other, CircuitStateFn) and self.is_measurement:
            from .. import Zero
            return self.compose(CircuitOp(other.primitive,
                                          other.coeff)).compose(Zero ^ self.num_qubits)

        from qiskit.aqua.operators import ComposedOp
        return ComposedOp([new_self, other])

    def tensor(self, other: OperatorBase) -> OperatorBase:
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
        # pylint: disable=import-outside-toplevel
        if isinstance(other, CircuitStateFn) and other.is_measurement == self.is_measurement:
            # Avoid reimplementing tensor, just use CircuitOp's
            from ..primitive_ops.circuit_op import CircuitOp
            from ..operator_globals import Zero
            c_op_self = CircuitOp(self.primitive, self.coeff)
            c_op_other = CircuitOp(other.primitive, other.coeff)
            return c_op_self.tensor(c_op_other).compose(Zero)
        # pylint: disable=cyclic-import
        from ..list_ops.tensored_op import TensoredOp
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

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # Rely on VectorStateFn's logic here.
        return StateFn(self.to_matrix() * self.coeff).to_density_matrix()

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # Need to adjoint to get forward statevector and then reverse
        if self.is_measurement:
            return np.conj(self.adjoint().to_matrix())
        qc = self.to_circuit(meas=False)
        statevector_backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(qc,
                              statevector_backend,
                              optimization_level=0).result().get_statevector()
        # pylint: disable=cyclic-import
        from ..operator_globals import EVAL_SIG_DIGITS
        return np.round(statevector * self.coeff, decimals=EVAL_SIG_DIGITS)

    def __str__(self) -> str:
        qc = self.reduce().to_circuit()  # type: ignore
        prim_str = str(qc.draw(output='text'))
        if self.coeff == 1.0:
            return "{}(\n{}\n)".format('CircuitStateFn' if not self.is_measurement
                                       else 'CircuitMeasurement', prim_str)
        else:
            return "{}(\n{}\n) * {}".format('CircuitStateFn' if not self.is_measurement
                                            else 'CircuitMeasurement',
                                            prim_str,
                                            self.coeff)

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        qc = self.primitive
        if isinstance(self.coeff, ParameterExpression) or self.primitive.parameters:
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..list_ops.list_op import ListOp
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if isinstance(self.coeff, ParameterExpression) \
                    and self.coeff.parameters <= set(unrolled_dict.keys()):
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

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        # pylint: disable=import-outside-toplevel
        from ..list_ops.list_op import ListOp
        from ..primitive_ops.pauli_op import PauliOp
        from ..primitive_ops.matrix_op import MatrixOp
        from ..primitive_ops.circuit_op import CircuitOp

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])

        # Composable with circuit
        if isinstance(front, (PauliOp, CircuitOp, MatrixOp, CircuitStateFn)):
            new_front = self.compose(front)
            return cast(Union[OperatorBase, float, complex], new_front.eval())

        return cast(Union[OperatorBase, float, complex], self.to_matrix_op().eval(front))

    def to_circuit(self, meas: bool = False) -> QuantumCircuit:
        """ Return QuantumCircuit representing StateFn """
        if meas:
            meas_qc = self.primitive.copy()
            meas_qc.add_register(ClassicalRegister(self.num_qubits))
            meas_qc.measure(qubit=range(self.num_qubits), cbit=range(self.num_qubits))
            return meas_qc
        else:
            return self.primitive

    def to_circuit_op(self) -> OperatorBase:
        """ Return ``StateFnCircuit`` corresponding to this StateFn."""
        return self

    def to_instruction(self):
        """ Return Instruction corresponding to primitive. """
        return self.primitive.to_instruction()

    # TODO specify backend?
    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        """
        Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability.
        """
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        qc = self.to_circuit(meas=True)
        qasm_backend = BasicAer.get_backend('qasm_simulator')
        counts = execute(qc, qasm_backend, optimization_level=0, shots=shots).result().get_counts()
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))

    # Warning - modifying primitive!!
    def reduce(self) -> OperatorBase:
        if self.primitive.data is not None:
            # Need to do this from the end because we're deleting items!
            for i in reversed(range(len(self.primitive.data))):
                [gate, _, _] = self.primitive.data[i]
                # Check if Identity or empty instruction (need to check that type is exactly
                # Instruction because some gates have lazy gate.definition population)
                # pylint: disable=unidiomatic-typecheck
                if isinstance(gate, IGate) or (type(gate) == Instruction and
                                               gate.definition.data == []):
                    del self.primitive.data[i]
        return self

    def permute(self, permutation: List[int]) -> 'CircuitStateFn':
        r"""
        Permute the qubits of the circuit.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j of the circuit should be permuted to position permutation[j].

        Returns:
            A new CircuitStateFn containing the permuted circuit.
        """
        new_qc = QuantumCircuit(self.num_qubits).compose(self.primitive, qubits=permutation)
        return CircuitStateFn(new_qc, coeff=self.coeff, is_measurement=self.is_measurement)
