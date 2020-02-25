# -*- coding: utf-8 -*-

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

"""
Arbitrary unitary circuit instruction.
"""

from collections import OrderedDict
import numpy

from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.extensions.standard import U3Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose
from qiskit.extensions.exceptions import ExtensionError

_DECOMPOSER1Q = OneQubitEulerDecomposer('U3')


class UnitaryGate(Gate):
    """Class for representing unitary gates"""

    def __init__(self, data, label=None):
        """Create a gate from a numeric unitary matrix.

        Args:
            data (matrix or Operator): unitary operator.
            label (str): unitary name for backend [Default: None].

        Raises:
            ExtensionError: if input data is not an N-qubit unitary operator.
        """
        if hasattr(data, 'to_matrix'):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, 'to_operator'):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data
        # Convert to numpy array in case not already an array
        data = numpy.array(data, dtype=complex)
        # Check input is unitary
        if not is_unitary_matrix(data):
            raise ExtensionError("Input matrix is not unitary.")
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        n_qubits = int(numpy.log2(input_dim))
        if input_dim != output_dim or 2**n_qubits != input_dim:
            raise ExtensionError(
                "Input matrix is not an N-qubit operator.")

        self._qasm_name = None
        self._qasm_definition = None
        self._qasm_def_written = False
        # Store instruction params
        super().__init__('unitary', n_qubits, [data], label=label)

    def __eq__(self, other):
        if not isinstance(other, UnitaryGate):
            return False
        if self.label != other.label:
            return False
        # Should we match unitaries as equal if they are equal
        # up to global phase?
        return matrix_equal(self.params[0], other.params[0], ignore_phase=True)

    def to_matrix(self):
        """Return matrix for the unitary."""
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
            theta, phi, lam = _DECOMPOSER1Q.angles(self.to_matrix())
            self.definition = [(U3Gate(theta, phi, lam), [q[0]], [])]
        elif self.num_qubits == 2:
            self.definition = two_qubit_cnot_decompose(self.to_matrix())
        else:
            raise NotImplementedError("Not able to generate a subcircuit for "
                                      "a {}-qubit unitary".format(self.num_qubits))

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        r"""Return controlled version of gate

        Args:
            num_ctrl_qubits (int): number of controls to add to gate (default=1)
            label (str): optional gate label
            ctrl_state (int or str or None): The control state in decimal or as a
                bit string (e.g. '1011'). If None, use 2**num_ctrl_qubits-1.

        Returns:
            UnitaryGate: controlled version of gate.

        Raises:
            QiskitError: invalid ctrl_state
        """
        cmat = _compute_control_matrix(self.to_matrix(), num_ctrl_qubits)
        return UnitaryGate(cmat, label=label)

    def qasm(self):
        """ The qasm for a custom unitary gate
        This is achieved by adding a custom gate that corresponds to the definition
        of this gate. It gives the gate a random name if one hasn't been given to it.
        """
        # if this is true then we have written the gate definition already
        # so we only need to write the name
        if self._qasm_def_written:
            return self._qasmif(self._qasm_name)

        # we have worked out the definition before, but haven't written it yet
        # so we need to write definition + name
        if self._qasm_definition:
            self._qasm_def_written = True
            return self._qasm_definition + self._qasmif(self._qasm_name)

        # need to work out the definition and then write it

        # give this unitary a name
        self._qasm_name = self.label if self.label else "unitary" + str(id(self))

        # map from gates in the definition to params in the method
        reg_to_qasm = OrderedDict()
        current_reg = 0

        gates_def = ""
        for gate in self.definition:

            # add regs from this gate to the overall set of params
            for reg in gate[1] + gate[2]:
                if reg not in reg_to_qasm:
                    reg_to_qasm[reg] = 'p' + str(current_reg)
                    current_reg += 1

            curr_gate = "\t%s %s;\n" % (gate[0].qasm(),
                                        ",".join([reg_to_qasm[j]
                                                  for j in gate[1] + gate[2]]))
            gates_def += curr_gate

        # name of gate + params + {definition}
        overall = "gate " + self._qasm_name + \
                  " " + ",".join(reg_to_qasm.values()) + \
                  " {\n" + gates_def + "}\n"

        self._qasm_def_written = True
        self._qasm_definition = overall

        return self._qasm_definition + self._qasmif(self._qasm_name)


def _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=None):
    r"""
    Compute the controlled version of the input matrix with qiskit ordering.
    This function computes the controlled unitary with :math:`n` control qubits
    and :math:`m` target qubits,

    .. math::

        V_n^j(U_{2^m}) = (U_{2^m} \otimes |j\rangle\!\langle j|) +
                         (I_{2^m} \otimes (I_{2^n} - |j\rangle\!\langle j|)).

    where :math:`|j\rangle \in \mathcal{H}^{2^n}` is the control state.

    Args:
        base_mat (ndarray): unitary to be controlled
        num_ctrl_qubits (int): number of controls for new unitary
        ctrl_state (int or str or None): The control state in decimal or as
            a bitstring (e.g. '111'). If None, use 2**num_ctrl_qubits-1.

    Returns:
        ndarray: controlled version of base matrix.

    Raises:
        QiskitError: unrecognized mode or invalid ctrl_state
    """
    num_target = int(numpy.log2(base_mat.shape[0]))
    ctrl_dim = 2**num_ctrl_qubits
    ctrl_grnd = numpy.repeat([[1], [0]], [1, ctrl_dim-1])
    if ctrl_state is None:
        ctrl_state = ctrl_dim - 1
    elif isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state, 2)
    if isinstance(ctrl_state, int):
        if not 0 <= ctrl_state < ctrl_dim:
            raise QiskitError('Invalid control state value specified.')
    else:
        raise QiskitError('Invalid control state type specified.')
    full_mat_dim = ctrl_dim * base_mat.shape[0]
    full_mat = numpy.zeros((full_mat_dim, full_mat_dim), dtype=base_mat.dtype)
    ctrl_proj = numpy.diag(numpy.roll(ctrl_grnd, ctrl_state))
    full_mat = (numpy.kron(numpy.eye(2**num_target),
                           numpy.eye(ctrl_dim) - ctrl_proj)
                + numpy.kron(base_mat, ctrl_proj))
    return full_mat


def unitary(self, obj, qubits, label=None):
    """Apply u2 to q."""
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]
    return self.append(UnitaryGate(obj, label=label), qubits, [])


QuantumCircuit.unitary = unitary
