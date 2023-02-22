# This code is part of Qiskit.
#
# (C) Copyright IBM 2017--2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit simulation for the Clifford class.
"""

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError


def _append_circuit(clifford, circuit, qargs=None):
    """Update Clifford inplace by applying a Clifford circuit.

    Args:
        clifford (Clifford): the Clifford to update.
        circuit (QuantumCircuit): the circuit to apply.
        qargs (list or None): The qubits to apply circuit to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input circuit cannot be decomposed into Clifford operations.
    """
    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    for instruction in circuit:
        if instruction.clbits:
            raise QiskitError(
                f"Cannot apply Instruction with classical bits: {instruction.operation.name}"
            )
        # Get the integer position of the flat register
        new_qubits = [qargs[circuit.find_bit(bit).index] for bit in instruction.qubits]
        _append_operation(clifford, instruction.operation, new_qubits)
    return clifford


def _append_operation(clifford, operation, qargs=None):
    """Update Clifford inplace by applying a Clifford operation.

    Args:
        clifford (Clifford): the Clifford to update.
        operation (Instruction or str): the operation or composite operation to apply.
        qargs (list or None): The qubits to apply operation to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input operation cannot be decomposed into Clifford operations.
    """
    if isinstance(operation, (Barrier, Delay)):
        return clifford

    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    gate = operation

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in _BASIS_1Q and gate not in _BASIS_2Q:
            raise QiskitError(f"Invalid Clifford gate name string {gate}")
        name = gate
    else:
        # assert isinstance(gate, Instruction)
        name = gate.name
        if getattr(gate, "condition", None) is not None:
            raise QiskitError("Conditional gate is not a valid Clifford operation.")

    # Apply gate if it is a Clifford basis gate
    if name in _NON_CLIFFORD:
        raise QiskitError(f"Cannot update Clifford with non-Clifford gate {name}")
    if name in _BASIS_1Q:
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate.")
        return _BASIS_1Q[name](clifford, qargs[0])
    if name in _BASIS_2Q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return _BASIS_2Q[name](clifford, qargs[0], qargs[1])

    # If gate is a Clifford, we can either unroll the gate using the "to_circuit"
    # method, or we can compose the Cliffords directly. Experimentally, for large
    # cliffords the second method is considerably faster.

    # pylint: disable=cyclic-import
    from qiskit.quantum_info import Clifford

    if isinstance(gate, Clifford):
        composed_clifford = clifford.compose(gate, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    # If not a Clifford basis gate we try to unroll the gate and
    # raise an exception if unrolling reaches a non-Clifford gate.
    # TODO: We could also check u3 params to see if they
    # are a single qubit Clifford gate rather than raise an exception.
    if gate.definition is None:
        raise QiskitError(f"Cannot apply Instruction: {gate.name}")

    return _append_circuit(clifford, gate.definition, qargs)


# ---------------------------------------------------------------------
# Helper functions for applying basis gates
# ---------------------------------------------------------------------


def _append_i(clifford, qubit):
    """Apply an I gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    # pylint: disable=unused-argument
    return clifford


def _append_x(clifford, qubit):
    """Apply an X gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.phase ^= clifford.z[:, qubit]
    return clifford


def _append_y(clifford, qubit):
    """Apply a Y gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x ^ z
    return clifford


def _append_z(clifford, qubit):
    """Apply an Z gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.phase ^= clifford.x[:, qubit]
    return clifford


def _append_h(clifford, qubit):
    """Apply a H gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    tmp = x.copy()
    x[:] = z
    z[:] = tmp
    return clifford


def _append_s(clifford, qubit):
    """Apply an S gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]

    clifford.phase ^= x & z
    z ^= x
    return clifford


def _append_sdg(clifford, qubit):
    """Apply an Sdg gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & ~z
    z ^= x
    return clifford


def _append_v(clifford, qubit):
    """Apply a V gate to a Clifford.

    This is equivalent to an Sdg gate followed by a H gate.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = x.copy()
    x ^= z
    z[:] = tmp
    return clifford


def _append_w(clifford, qubit):
    """Apply a W gate to a Clifford.

    This is equivalent to two V gates.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = z.copy()
    z ^= x
    x[:] = tmp
    return clifford


def _append_cx(clifford, control, target):
    """Apply a CX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1
    return clifford


def _append_cz(clifford, control, target):
    """Apply a CZ gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= x0 & x1 & (z0 ^ z1)
    z1 ^= x0
    z0 ^= x1
    return clifford


def _append_swap(clifford, qubit0, qubit1):
    """Apply a Swap gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.x[:, [qubit0, qubit1]] = clifford.x[:, [qubit1, qubit0]]
    clifford.z[:, [qubit0, qubit1]] = clifford.z[:, [qubit1, qubit0]]
    return clifford


# Basis Clifford Gates
_BASIS_1Q = {
    "i": _append_i,
    "id": _append_i,
    "iden": _append_i,
    "x": _append_x,
    "y": _append_y,
    "z": _append_z,
    "h": _append_h,
    "s": _append_s,
    "sdg": _append_sdg,
    "sinv": _append_sdg,
    "v": _append_v,
    "w": _append_w,
}
_BASIS_2Q = {"cx": _append_cx, "cz": _append_cz, "swap": _append_swap}
# Non-clifford gates
_NON_CLIFFORD = {"t", "tdg", "ccx", "ccz"}
