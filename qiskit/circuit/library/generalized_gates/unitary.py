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

"""Arbitrary unitary circuit instruction."""

from __future__ import annotations

import typing
import numpy

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

# pylint: disable=cyclic-import
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose

from .isometry import Isometry

_DECOMPOSER1Q = OneQubitEulerDecomposer("U")

if typing.TYPE_CHECKING:
    from qiskit.quantum_info.operators.base_operator import BaseOperator


class UnitaryGate(Gate):
    """Class quantum gates specified by a unitary matrix.

    Example:

        We can create a unitary gate from a unitary matrix then add it to a
        quantum circuit. The matrix can also be directly applied to the quantum
        circuit, see :meth:`.QuantumCircuit.unitary`.

        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.circuit.library import UnitaryGate

            matrix = [[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]]
            gate = UnitaryGate(matrix)

            circuit = QuantumCircuit(2)
            circuit.append(gate, [0, 1])
    """

    def __init__(
        self,
        data: numpy.ndarray | Gate | BaseOperator,
        label: str | None = None,
        check_input: bool = True,
    ) -> None:
        """Create a gate from a numeric unitary matrix.

        Args:
            data: Unitary operator.
            label: Unitary name for backend [Default: None].
            check_input: If set to ``False`` this asserts the input
                is known to be unitary and the checking to validate this will
                be skipped. This should only ever be used if you know the
                input is unitary, setting this to ``False`` and passing in
                a non-unitary matrix will result unexpected behavior and errors.

        Raises:
            ValueError: If input data is not an N-qubit unitary operator.
        """
        if hasattr(data, "to_matrix"):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, "to_operator"):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data
        # Convert to numpy array in case not already an array
        data = numpy.asarray(data, dtype=complex)
        input_dim, output_dim = data.shape
        num_qubits = int(numpy.log2(input_dim))
        if check_input:
            # Check input is unitary
            if not is_unitary_matrix(data):
                raise ValueError("Input matrix is not unitary.")
            # Check input is N-qubit matrix
            if input_dim != output_dim or 2**num_qubits != input_dim:
                raise ValueError("Input matrix is not an N-qubit operator.")
        # Store instruction params
        super().__init__("unitary", num_qubits, [data], label=label)

    def __eq__(self, other):
        if not isinstance(other, UnitaryGate):
            return False
        if self.label != other.label:
            return False
        # Should we match unitaries as equal if they are equal
        # up to global phase?
        return matrix_equal(self.params[0], other.params[0], ignore_phase=True)

    def __array__(self, dtype=None):
        """Return matrix for the unitary."""
        # pylint: disable=unused-argument
        return self.params[0]

    def inverse(self):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the conjugate of the unitary."""
        return UnitaryGate(numpy.conj(self.to_matrix()))

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return self.transpose().conjugate()

    def transpose(self):
        """Return the transpose of the unitary."""
        return UnitaryGate(numpy.transpose(self.to_matrix()))

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        if self.num_qubits == 1:
            q = QuantumRegister(1, "q")
            qc = QuantumCircuit(q, name=self.name)
            theta, phi, lam, global_phase = _DECOMPOSER1Q.angles_and_phase(self.to_matrix())
            qc._append(UGate(theta, phi, lam), [q[0]], [])
            qc.global_phase = global_phase
            self.definition = qc
        elif self.num_qubits == 2:
            self.definition = two_qubit_cnot_decompose(self.to_matrix())
        else:
            from qiskit.quantum_info.synthesis.qsd import (  # pylint: disable=cyclic-import
                qs_decomposition,
            )

            self.definition = qs_decomposition(self.to_matrix())

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: int | None = None,
        ctrl_state: int | str | None = None,
    ) -> ControlledGate:
        """Return controlled version of gate.

        Args:
            num_ctrl_qubits: Number of controls to add to gate (default is 1).
            label: Optional gate label.
            ctrl_state: The control state in decimal or as a bit string (e.g. ``"1011"``).
                If ``None``, use ``2**num_ctrl_qubits - 1``.

        Returns:
            Controlled version of gate.
        """
        mat = self.to_matrix()
        cmat = _compute_control_matrix(mat, num_ctrl_qubits, ctrl_state=None)
        iso = Isometry(cmat, 0, 0)
        return ControlledGate(
            "c-unitary",
            num_qubits=self.num_qubits + num_ctrl_qubits,
            params=[mat],
            label=label,
            num_ctrl_qubits=num_ctrl_qubits,
            definition=iso.definition,
            ctrl_state=ctrl_state,
            base_gate=self.copy(),
        )

    def _qasm2_decomposition(self):
        """Return an unparameterized version of ourselves, so the OQ2 exporter doesn't choke on the
        non-standard things in our `params` field."""
        out = self.definition.to_gate()
        out.name = self.name
        return out

    def validate_parameter(self, parameter):
        """Unitary gate parameter has to be an ndarray."""
        if isinstance(parameter, numpy.ndarray):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} in gate {self.name}")
