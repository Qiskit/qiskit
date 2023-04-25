# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-param-doc
# pylint: disable=missing-type-doc

"""
Generic isometries from m to n qubits.
"""

import itertools
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from qiskit.extensions.quantum_initializer.uc import UCGate
from qiskit.extensions.quantum_initializer.mcg_up_to_diagonal import MCGupDiag

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Isometry(Instruction):
    """
    Decomposition of arbitrary isometries from m to n qubits. In particular, this allows to
    decompose unitaries (m=n) and to do state preparation (m=0).

    The decomposition is based on https://arxiv.org/abs/1501.06911.

    Args:
        isometry (ndarray): an isometry from m to n qubits, i.e., a (complex)
            np.ndarray of dimension 2^n*2^m with orthonormal columns (given
            in the computational basis specified by the order of the ancillas
            and the input qubits, where the ancillas are considered to be more
            significant than the input qubits).

        num_ancillas_zero (int): number of additional ancillas that start in the state ket(0)
            (the n-m ancillas required for providing the output of the isometry are
            not accounted for here).

        num_ancillas_dirty (int): number of additional ancillas that start in an arbitrary state

        epsilon (float) (optional): error tolerance of calculations
    """

    # Notation: In the following decomposition we label the qubit by
    # 0 -> most significant one
    # ...
    # n -> least significant one
    # finally, we convert the labels back to the qubit numbering used in Qiskit
    # (using: _get_qubits_by_label)

    def __init__(self, isometry, num_ancillas_zero, num_ancillas_dirty, epsilon=_EPS):
        # Convert to numpy array in case not already an array
        isometry = np.array(isometry, dtype=complex)

        # change a row vector to a column vector (in the case of state preparation)
        if len(isometry.shape) == 1:
            isometry = isometry.reshape(isometry.shape[0], 1)

        self.iso_data = isometry

        self.num_ancillas_zero = num_ancillas_zero
        self.num_ancillas_dirty = num_ancillas_dirty
        self._inverse = None
        self._epsilon = epsilon

        # Check if the isometry has the right dimension and if the columns are orthonormal
        n = np.log2(isometry.shape[0])
        m = np.log2(isometry.shape[1])
        if not n.is_integer() or n < 0:
            raise QiskitError(
                "The number of rows of the isometry is not a non negative power of 2."
            )
        if not m.is_integer() or m < 0:
            raise QiskitError(
                "The number of columns of the isometry is not a non negative power of 2."
            )
        if m > n:
            raise QiskitError(
                "The input matrix has more columns than rows and hence it can't be an isometry."
            )
        if not is_isometry(isometry, self._epsilon):
            raise QiskitError(
                "The input matrix has non orthonormal columns and hence it is not an isometry."
            )

        num_qubits = int(n) + num_ancillas_zero + num_ancillas_dirty

        super().__init__("isometry", num_qubits, 0, [isometry])

    def _define(self):

        # TODO The inverse().inverse() is because there is code to uncompute (_gates_to_uncompute)
        #  an isometry, but not for generating its decomposition. It would be cheaper to do the
        #  later here instead.
        gate = self.inv_gate()
        gate = gate.inverse()
        q = QuantumRegister(self.num_qubits)
        iso_circuit = QuantumCircuit(q)
        iso_circuit.append(gate, q[:])
        self.definition = iso_circuit

    def inverse(self):
        self.params = []
        inv = super().inverse()
        self.params = [self.iso_data]
        return inv

    def _gates_to_uncompute(self):
        """
        Call to create a circuit with gates that take the desired isometry to the first 2^m columns
         of the 2^n*2^n identity matrix (see https://arxiv.org/abs/1501.06911)
        """
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        (
            q_input,
            q_ancillas_for_output,
            q_ancillas_zero,
            q_ancillas_dirty,
        ) = self._define_qubit_role(q)
        # Copy the isometry (this is computationally expensive for large isometries but guarantees
        # to keep a copyof the input isometry)
        remaining_isometry = self.iso_data.astype(complex)  # note: "astype" does copy the isometry
        diag = []
        m = int(np.log2((self.iso_data).shape[1]))
        # Decompose the column with index column_index and attache the gate to the circuit object.
        # Return the isometry that is left to decompose, where the columns up to index column_index
        # correspond to the firstfew columns of the identity matrix up to diag, and hence we only
        # have to save a list containing them.
        for column_index in range(2**m):
            self._decompose_column(circuit, q, diag, remaining_isometry, column_index)
            # extract phase of the state that was sent to the basis state ket(column_index)
            diag.append(remaining_isometry[column_index, 0])
            # remove first column (which is now stored in diag)
            remaining_isometry = remaining_isometry[:, 1:]
        if len(diag) > 1 and not _diag_is_identity_up_to_global_phase(diag, self._epsilon):
            circuit.diagonal(np.conj(diag).tolist(), q_input)
        return circuit

    def _decompose_column(self, circuit, q, diag, remaining_isometry, column_index):
        """
        Decomposes the column with index column_index.
        """
        n = int(np.log2(self.iso_data.shape[0]))
        for s in range(n):
            self._disentangle(circuit, q, diag, remaining_isometry, column_index, s)

    def _disentangle(self, circuit, q, diag, remaining_isometry, column_index, s):
        """
        Disentangle the s-th significant qubit (starting with s = 0) into the zero or the one state
        (dependent on column_index)
        """
        # To shorten the notation, we introduce:
        k = column_index
        # k_prime is the index of the column with index column_index in the remaining isometry
        # (note that we remove columns of the isometry during the procedure for efficiency)
        k_prime = 0
        v = remaining_isometry
        n = int(np.log2(self.iso_data.shape[0]))

        # MCG to set one entry to zero (preparation for disentangling with UCGate):
        index1 = 2 * _a(k, s + 1) * 2**s + _b(k, s + 1)
        index2 = (2 * _a(k, s + 1) + 1) * 2**s + _b(k, s + 1)
        target_label = n - s - 1
        # Check if a MCG is required
        if _k_s(k, s) == 0 and _b(k, s + 1) != 0 and np.abs(v[index2, k_prime]) > self._epsilon:
            # Find the MCG, decompose it and apply it to the remaining isometry
            gate = _reverse_qubit_state([v[index1, k_prime], v[index2, k_prime]], 0, self._epsilon)
            control_labels = [
                i
                for i, x in enumerate(_get_binary_rep_as_list(k, n))
                if x == 1 and i != target_label
            ]
            diagonal_mcg = self._append_mcg_up_to_diagonal(
                circuit, q, gate, control_labels, target_label
            )
            # apply the MCG to the remaining isometry
            _apply_multi_controlled_gate(v, control_labels, target_label, gate)
            # correct for the implementation "up to diagonal"
            diag_mcg_inverse = np.conj(diagonal_mcg).tolist()
            _apply_diagonal_gate(v, control_labels + [target_label], diag_mcg_inverse)
            # update the diag according to the applied diagonal gate
            _apply_diagonal_gate_to_diag(diag, control_labels + [target_label], diag_mcg_inverse, n)

        # UCGate to disentangle a qubit:
        # Find the UCGate, decompose it and apply it to the remaining isometry
        single_qubit_gates = self._find_squs_for_disentangling(v, k, s)
        if not _ucg_is_identity_up_to_global_phase(single_qubit_gates, self._epsilon):
            control_labels = list(range(target_label))
            diagonal_ucg = self._append_ucg_up_to_diagonal(
                circuit, q, single_qubit_gates, control_labels, target_label
            )
            # merge the diagonal into the UCGate for efficient application of both together
            diagonal_ucg_inverse = np.conj(diagonal_ucg).tolist()
            single_qubit_gates = _merge_UCGate_and_diag(single_qubit_gates, diagonal_ucg_inverse)
            # apply the UCGate (with the merged diagonal gate) to the remaining isometry
            _apply_ucg(v, len(control_labels), single_qubit_gates)
            # update the diag according to the applied diagonal gate
            _apply_diagonal_gate_to_diag(
                diag, control_labels + [target_label], diagonal_ucg_inverse, n
            )
            # # correct for the implementation "up to diagonal"
            # diag_inv = np.conj(diag).tolist()
            # _apply_diagonal_gate(v, control_labels + [target_label], diag_inv)

    # This method finds the single-qubit gates for a UCGate to disentangle a qubit:
    # we consider the n-qubit state v[:,0] starting with k zeros (in the computational basis).
    # The qubit with label n-s-1 is disentangled into the basis state k_s(k,s).
    def _find_squs_for_disentangling(self, v, k, s):
        k_prime = 0
        n = int(np.log2(self.iso_data.shape[0]))
        if _b(k, s + 1) == 0:
            i_start = _a(k, s + 1)
        else:
            i_start = _a(k, s + 1) + 1
        id_list = [np.eye(2, 2) for _ in range(i_start)]
        squs = [
            _reverse_qubit_state(
                [
                    v[2 * i * 2**s + _b(k, s), k_prime],
                    v[(2 * i + 1) * 2**s + _b(k, s), k_prime],
                ],
                _k_s(k, s),
                self._epsilon,
            )
            for i in range(i_start, 2 ** (n - s - 1))
        ]
        return id_list + squs

    # Append a UCGate up to diagonal to the circuit circ.
    def _append_ucg_up_to_diagonal(self, circ, q, single_qubit_gates, control_labels, target_label):
        (
            q_input,
            q_ancillas_for_output,
            q_ancillas_zero,
            q_ancillas_dirty,
        ) = self._define_qubit_role(q)
        n = int(np.log2(self.iso_data.shape[0]))
        qubits = q_input + q_ancillas_for_output
        # Note that we have to reverse the control labels, since controls are provided by
        # increasing qubit number toa UCGate by convention
        control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
        target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
        ucg = UCGate(single_qubit_gates, up_to_diagonal=True)
        circ.append(ucg, [target_qubit] + control_qubits)
        return ucg._get_diagonal()

    # Append a MCG up to diagonal to the circuit circ. The diagonal should only act on the control
    # and target qubits and not on the ancillas. In principle, it would be allowed to act on the
    # dirty ancillas on which we perform the isometry (i.e., on the qubits listed in "qubits"
    # below). But for simplicity, the current code version ignores this future optimization
    # possibility.
    def _append_mcg_up_to_diagonal(self, circ, q, gate, control_labels, target_label):
        (
            q_input,
            q_ancillas_for_output,
            q_ancillas_zero,
            q_ancillas_dirty,
        ) = self._define_qubit_role(q)
        n = int(np.log2(self.iso_data.shape[0]))
        qubits = q_input + q_ancillas_for_output
        control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
        target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
        # The qubits on which we neither act nor control on with the MCG, can be used
        # as dirty ancillas
        ancilla_dirty_labels = [i for i in range(n) if i not in control_labels + [target_label]]
        ancillas_dirty = (
            _reverse_qubit_oder(_get_qubits_by_label(ancilla_dirty_labels, qubits, n))
            + q_ancillas_dirty
        )
        mcg_up_to_diag = MCGupDiag(
            gate, len(control_qubits), len(q_ancillas_zero), len(ancillas_dirty)
        )
        circ.append(
            mcg_up_to_diag, [target_qubit] + control_qubits + q_ancillas_zero + ancillas_dirty
        )
        return mcg_up_to_diag._get_diagonal()

    def _define_qubit_role(self, q):

        n = int(np.log2(self.iso_data.shape[0]))
        m = int(np.log2(self.iso_data.shape[1]))

        # Define the role of the qubits
        q_input = q[:m]
        q_ancillas_for_output = q[m:n]
        q_ancillas_zero = q[n : n + self.num_ancillas_zero]
        q_ancillas_dirty = q[n + self.num_ancillas_zero :]
        return q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty

    def validate_parameter(self, parameter):
        """Isometry parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        if isinstance(parameter, (list, int)):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} for gate {self.name}")

    def inv_gate(self):
        """Return the adjoint of the unitary."""
        if self._inverse is None:
            # call to generate the circuit that takes the isometry to the first 2^m columns
            # of the 2^n identity matrix
            iso_circuit = self._gates_to_uncompute()
            # invert the circuit to create the circuit implementing the isometry
            self._inverse = iso_circuit.to_instruction()
        return self._inverse


# Find special unitary matrix that maps [c0,c1] to [r,0] or [0,r] if basis_state=0 or
# basis_state=1 respectively
def _reverse_qubit_state(state, basis_state, epsilon):
    state = np.array(state)
    r = np.linalg.norm(state)
    if r < epsilon:
        return np.eye(2, 2)
    if basis_state == 0:
        m = np.array([[np.conj(state[0]), np.conj(state[1])], [-state[1], state[0]]]) / r
    else:
        m = np.array([[-state[1], state[0]], [np.conj(state[0]), np.conj(state[1])]]) / r
    return m


# Methods for applying gates to matrices (should be moved to Qiskit AER)

# Input: matrix m with 2^n rows (and arbitrary many columns). Think of the columns as states
#  on n qubits. The method applies a uniformly controlled gate (UCGate) to all the columns, where
#  the UCGate is specified by the inputs k and single_qubit_gates:

#  k =  number of controls. We assume that the controls are on the k most significant qubits
#       (and the target is on the (k+1)th significant qubit)
#  single_qubit_gates =     [u_0,...,u_{2^k-1}], where the u_i's are 2*2 unitaries
#                           (provided as numpy arrays)

# The order of the single-qubit unitaries is such that the first unitary u_0 is applied to the
# (k+1)th significant qubit if the control qubits are in the state ket(0...00), the gate u_1 is
# applied if the control qubits are in the state ket(0...01), and so on.

# The input matrix m and the single-qubit gates have to be of dtype=complex.


def _apply_ucg(m, k, single_qubit_gates):
    # ToDo: Improve efficiency by parallelizing the gate application. A generalized version of
    # ToDo: this method should be implemented by the state vector simulator in Qiskit AER.
    num_qubits = int(np.log2(m.shape[0]))
    num_col = m.shape[1]
    spacing = 2 ** (num_qubits - k - 1)
    for j in range(2 ** (num_qubits - 1)):
        i = (j // spacing) * spacing + j
        gate_index = i // (2 ** (num_qubits - k))
        for col in range(num_col):
            m[np.array([i, i + spacing]), np.array([col, col])] = np.ndarray.flatten(
                single_qubit_gates[gate_index].dot(np.array([[m[i, col]], [m[i + spacing, col]]]))
            ).tolist()
    return m


# Apply a diagonal gate with diagonal entries liste in diag and acting on qubits with labels
#  action_qubit_labels to a matrix m.
# The input matrix m has to be of dtype=complex
# The qubit labels are such that label 0 corresponds to the most significant qubit, label 1 to
#  the second most significant qubit, and so on ...


def _apply_diagonal_gate(m, action_qubit_labels, diag):
    # ToDo: Improve efficiency by parallelizing the gate application. A generalized version of
    # ToDo: this method should be implemented by the state vector simulator in Qiskit AER.
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    basis_states = list(itertools.product([0, 1], repeat=num_qubits))
    for state in basis_states:
        state_on_action_qubits = [state[i] for i in action_qubit_labels]
        diag_index = _bin_to_int(state_on_action_qubits)
        i = _bin_to_int(state)
        for j in range(num_cols):
            m[i, j] = diag[diag_index] * m[i, j]
    return m


# Special case of the method _apply_diagonal_gate, where the input m is a diagonal matrix on the
# log2(len(m_diagonal)) least significant qubits (this method is more efficient in this case
# than _apply_diagonal_gate). The input m_diagonal is provided as a list of diagonal entries.
# The diagonal diag is applied on the qubits with labels listed in action_qubit_labels. The input
# num_qubits gives the total number of considered qubits (this input is required to interpret the
# action_qubit_labels in relation to the least significant qubits).


def _apply_diagonal_gate_to_diag(m_diagonal, action_qubit_labels, diag, num_qubits):
    if not m_diagonal:
        return m_diagonal
    basis_states = list(itertools.product([0, 1], repeat=num_qubits))
    for state in basis_states[: len(m_diagonal)]:
        state_on_action_qubits = [state[i] for i in action_qubit_labels]
        diag_index = _bin_to_int(state_on_action_qubits)
        i = _bin_to_int(state)
        m_diagonal[i] *= diag[diag_index]
    return m_diagonal


# Apply a MC single-qubit gate (given by the 2*2 unitary input: gate) with controlling on
# the qubits with label control_labels and acting on the qubit with label target_label
# to a matrix m. The input matrix m and the gate have to be of dtype=complex. The qubit labels are
# such that label 0 corresponds to the most significant qubit, label 1 to the second most
# significant qubit, and so on ...


def _apply_multi_controlled_gate(m, control_labels, target_label, gate):
    # ToDo: This method should be integrated into the state vector simulator in Qiskit AER.
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    control_labels.sort()
    free_qubits = num_qubits - len(control_labels) - 1
    basis_states_free = list(itertools.product([0, 1], repeat=free_qubits))
    for state_free in basis_states_free:
        (e1, e2) = _construct_basis_states(state_free, control_labels, target_label)
        for i in range(num_cols):
            m[np.array([e1, e2]), np.array([i, i])] = np.ndarray.flatten(
                gate.dot(np.array([[m[e1, i]], [m[e2, i]]]))
            ).tolist()
    return m


# Helper method for _apply_multi_controlled_gate. This constructs the basis states the MG gate
# is acting on for a specific state state_free of the qubits we neither control nor act on.


def _construct_basis_states(state_free, control_labels, target_label):
    e1 = []
    e2 = []
    j = 0
    for i in range(len(state_free) + len(control_labels) + 1):
        if i in control_labels:
            e1.append(1)
            e2.append(1)
        elif i == target_label:
            e1.append(0)
            e2.append(1)
        else:
            e1.append(state_free[j])
            e2.append(state_free[j])
            j += 1
    out1 = _bin_to_int(e1)
    out2 = _bin_to_int(e2)
    return out1, out2


# Some helper methods:


# Get the qubits in the list qubits corresponding to the labels listed in labels. The total number
# of qubits is given by num_qubits (and determines the convention for the qubit labeling)

# Remark: We labeled the qubits with decreasing significance. So we have to transform the labels to
# be compatible with the standard convention of Qiskit.


def _get_qubits_by_label(labels, qubits, num_qubits):
    return [qubits[num_qubits - label - 1] for label in labels]


def _reverse_qubit_oder(qubits):
    return list(reversed(qubits))


# Convert list of binary digits to integer


def _bin_to_int(binary_digits_as_list):
    return int("".join(str(x) for x in binary_digits_as_list), 2)


def _ct(m):
    return np.transpose(np.conjugate(m))


def _get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary[-num_digits:]


# absorb a diagonal gate into a UCGate


def _merge_UCGate_and_diag(single_qubit_gates, diag):
    for (i, gate) in enumerate(single_qubit_gates):
        single_qubit_gates[i] = np.array([[diag[2 * i], 0.0], [0.0, diag[2 * i + 1]]]).dot(gate)
    return single_qubit_gates


# Helper variables/functions for the column-by-column decomposition


# a(k,s) and b(k,s) are positive integers such that k = a(k,s)2^s + b(k,s)
# (with the maximal choice of a(k,s))


def _a(k, s):
    return k // 2**s


def _b(k, s):
    return k - (_a(k, s) * 2**s)


# given a binary representation of k with binary digits [k_{n-1},..,k_1,k_0],
# the method k_s(k, s) returns k_s


def _k_s(k, s):
    if k == 0:
        return 0
    else:
        num_digits = s + 1
        return _get_binary_rep_as_list(k, num_digits)[0]


# Check if a gate of a special form is equal to the identity gate up to global phase


def _ucg_is_identity_up_to_global_phase(single_qubit_gates, epsilon):
    if not np.abs(single_qubit_gates[0][0, 0]) < epsilon:
        global_phase = 1.0 / (single_qubit_gates[0][0, 0])
    else:
        return False
    for gate in single_qubit_gates:
        if not np.allclose(global_phase * gate, np.eye(2, 2)):
            return False
    return True


def _diag_is_identity_up_to_global_phase(diag, epsilon):
    if not np.abs(diag[0]) < epsilon:
        global_phase = 1.0 / (diag[0])
    else:
        return False
    for d in diag:
        if not np.abs(global_phase * d - 1) < epsilon:
            return False
    return True


def iso(
    self,
    isometry,
    q_input,
    q_ancillas_for_output,
    q_ancillas_zero=None,
    q_ancillas_dirty=None,
    epsilon=_EPS,
):
    """
    Attach an arbitrary isometry from m to n qubits to a circuit. In particular,
    this allows to attach arbitrary unitaries on n qubits (m=n) or to prepare any state
    on n qubits (m=0).
    The decomposition used here was introduced by Iten et al. in https://arxiv.org/abs/1501.06911.

    Args:
        isometry (ndarray): an isometry from m to n qubits, i.e., a (complex) ndarray of
            dimension 2^n×2^m with orthonormal columns (given in the computational basis
            specified by the order of the ancillas and the input qubits, where the ancillas
            are considered to be more significant than the input qubits.).
        q_input (QuantumRegister|list[Qubit]): list of m qubits where the input
            to the isometry is fed in (empty list for state preparation).
        q_ancillas_for_output (QuantumRegister|list[Qubit]): list of n-m ancilla
            qubits that are used for the output of the isometry and which are assumed to start
            in the zero state. The qubits are listed with increasing significance.
        q_ancillas_zero (QuantumRegister|list[Qubit]): list of ancilla qubits
            which are assumed to start in the zero state. Default is q_ancillas_zero = None.
        q_ancillas_dirty (QuantumRegister|list[Qubit]): list of ancilla qubits
            which can start in an arbitrary state. Default is q_ancillas_dirty = None.
        epsilon (float): error tolerance of calculations.
            Default is epsilon = _EPS.

    Returns:
        QuantumCircuit: the isometry is attached to the quantum circuit.

    Raises:
        QiskitError: if the array is not an isometry of the correct size corresponding to
            the provided number of qubits.
    """
    if q_input is None:
        q_input = []
    if q_ancillas_for_output is None:
        q_ancillas_for_output = []
    if q_ancillas_zero is None:
        q_ancillas_zero = []
    if q_ancillas_dirty is None:
        q_ancillas_dirty = []

    if isinstance(q_input, QuantumRegister):
        q_input = q_input[:]
    if isinstance(q_ancillas_for_output, QuantumRegister):
        q_ancillas_for_output = q_ancillas_for_output[:]
    if isinstance(q_ancillas_zero, QuantumRegister):
        q_ancillas_zero = q_ancillas_zero[:]
    if isinstance(q_ancillas_dirty, QuantumRegister):
        q_ancillas_dirty = q_ancillas_dirty[:]

    return self.append(
        Isometry(isometry, len(q_ancillas_zero), len(q_ancillas_dirty), epsilon=epsilon),
        q_input + q_ancillas_for_output + q_ancillas_zero + q_ancillas_dirty,
    )


# support both QuantumCircuit.iso and QuantumCircuit.isometry
QuantumCircuit.iso = iso
QuantumCircuit.isometry = iso
