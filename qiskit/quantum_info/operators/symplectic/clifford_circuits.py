# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
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

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier


def _append_circuit(clifford, circuit, qargs=None):
    """Update Clifford inplace by applying a Clifford circuit.

    Args:
        clifford (Clifford): the Clifford to update.
        circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gate to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input gate cannot be decomposed into Clifford gates.
    """
    if isinstance(circuit, Barrier):
        return clifford

    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    if isinstance(circuit, QuantumCircuit):
        gate = circuit.to_instruction()
    else:
        gate = circuit

    # Basis Clifford Gates
    basis_1q = {
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
    basis_2q = {"cx": _append_cx, "cz": _append_cz, "swap": _append_swap}

    # Non-clifford gates
    non_clifford = ["t", "tdg", "ccx", "ccz"]

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in basis_1q and gate not in basis_2q:
            raise QiskitError(f"Invalid Clifford gate name string {gate}")
        name = gate
    else:
        # Assume gate is an Instruction
        name = gate.name

    # Apply gate if it is a Clifford basis gate
    if name in non_clifford:
        raise QiskitError(f"Cannot update Clifford with non-Clifford gate {name}")
    if name in basis_1q:
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate.")
        return basis_1q[name](clifford, qargs[0])
    if name in basis_2q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return basis_2q[name](clifford, qargs[0], qargs[1])

    # If not a Clifford basis gate we try to unroll the gate and
    # raise an exception if unrolling reaches a non-Clifford gate.
    # TODO: We could also check u3 params to see if they
    # are a single qubit Clifford gate rather than raise an exception.
    if gate.definition is None:
        raise QiskitError(f"Cannot apply Instruction: {gate.name}")
    if not isinstance(gate.definition, QuantumCircuit):
        raise QiskitError(
            "{} instruction definition is {}; expected QuantumCircuit".format(
                gate.name, type(gate.definition)
            )
        )
    qubit_indices = {bit: idx for idx, bit in enumerate(gate.definition.qubits)}
    for instr, qregs, cregs in gate.definition:
        if cregs:
            raise QiskitError(f"Cannot apply Instruction with classical registers: {instr.name}")
        # Get the integer position of the flat register
        new_qubits = [qargs[qubit_indices[tup]] for tup in qregs]
        _append_circuit(clifford, instr, new_qubits)
    return clifford


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
    clifford.table.phase ^= clifford.table.Z[:, qubit]
    return clifford


def _append_y(clifford, qubit):
    """Apply a Y gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= x ^ z
    return clifford


def _append_z(clifford, qubit):
    """Apply an Z gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.table.phase ^= clifford.table.X[:, qubit]
    return clifford


def _append_h(clifford, qubit):
    """Apply a H gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= x & z
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
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]

    clifford.table.phase ^= x & z
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
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
    clifford.table.phase ^= x & ~z
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
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
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
    x = clifford.table.X[:, qubit]
    z = clifford.table.Z[:, qubit]
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
    x0 = clifford.table.X[:, control]
    z0 = clifford.table.Z[:, control]
    x1 = clifford.table.X[:, target]
    z1 = clifford.table.Z[:, target]
    clifford.table.phase ^= (x1 ^ z0 ^ True) & z1 & x0
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
    x0 = clifford.table.X[:, control]
    z0 = clifford.table.Z[:, control]
    x1 = clifford.table.X[:, target]
    z1 = clifford.table.Z[:, target]
    clifford.table.phase ^= x0 & x1 & (z0 ^ z1)
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
    clifford.table.X[:, [qubit0, qubit1]] = clifford.table.X[:, [qubit1, qubit0]]
    clifford.table.Z[:, [qubit0, qubit1]] = clifford.table.Z[:, [qubit1, qubit0]]
    return clifford
