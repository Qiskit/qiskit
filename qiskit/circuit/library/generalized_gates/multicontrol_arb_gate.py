# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""multicontrol single qubit unitary gate."""

from typing import List
from numpy import array, pi, allclose, eye
from scipy.linalg import fractional_matrix_power
from qiskit.circuit import QuantumCircuit


def multiconrol_single_qubit_gate(
    single_q_unitary: array, control_list: List[int], target_q: int
) -> QuantumCircuit:
    """
    Generate a quantum circuit to implement a defined multicontrol single qubit unitary
    via approach proposed in  https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        single_q_unitary (array): two by two unitary matrix to implement as a control operation
        control_list (list): list of control qubit indices
        target_q (int): target qubit index
    Returns:
        circuit (QuantumCircuit): quantum circuit implementing multicontrol single_q_unitary

    """
    assert target_q not in control_list, f"target qubit: {target_q} in control list"

    n_qubits = max(*control_list, target_q) + 1
    circuit = QuantumCircuit(n_qubits)
    cnu_gate = cn_u(len(control_list), single_q_unitary).to_gate()
    circuit.append(cnu_gate, [*control_list, target_q])
    return circuit


def cn_u(n_controls: int, single_q_unitary: array) -> QuantumCircuit:
    """
    Implement a multicontrol U gate according to https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        n_controls(int): number of control qubits
        single_q_unitary (array): two by two unitary matrix to implement as a control operation
    Returns:
        circuit (QuantumCircuit): Quantum circuit implementing multicontrol unitary
    """
    assert single_q_unitary.shape == (2, 2), "input unitary is not a single qubit gate"
    assert allclose(single_q_unitary @ single_q_unitary.conj().T, eye(2)), "input unitary is not unitary"

    targ = n_controls + 1
    circuit = QuantumCircuit(targ)
    if n_controls == 1:
        ucirc = QuantumCircuit(1)
        ucirc.unitary(single_q_unitary, 0)
        ucirc.name = "U"
        ucircgate = ucirc.to_gate().control(1)
        circuit.append(ucircgate, [0, n_controls])
    else:
        pnu = pnu_gate(n_controls, single_q_unitary)
        circuit = circuit.compose(pnu)

        power = n_controls - 1
        rootu = fractional_matrix_power(single_q_unitary, 1 / 2 ** (n_controls - 1))
        rootucirc = QuantumCircuit(1)
        rootucirc.unitary(rootu, 0)
        rootucirc.name = f"U^1/{2 ** power}"
        controlrootugate = rootucirc.to_gate().control(1)
        circuit.append(controlrootugate, [0, n_controls])

        qn = qn_gate(n_controls)
        circuit = circuit.compose(qn)
        circuit = circuit.compose(pnu.inverse())
        circuit = circuit.compose(qn.inverse())

    return circuit


def pn_gate(n_controls: int) -> QuantumCircuit:
    """
    Pn gate defined in equation 1 of https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        n_controls (int): number of controls
    Returns:
        circuit (QuantumCircuit): quantum circuit of Pn gate
    """
    assert n_controls > 0, "number of controls must be 1 or more!"

    # target = n_controls[-1] +1 for now!
    circuit = QuantumCircuit(n_controls + 1)

    for k in reversed(range(2, n_controls + 1)):
        circuit.crx(pi / 2 ** (n_controls - k + 1), k - 1, n_controls)

    return circuit


def pnu_gate(n_controls: int, single_q_unitary: array) -> QuantumCircuit:
    """
    Pn(U) gate defined in equation 2 of https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        n_controls (int): number of controls
        single_q_unitary (array): two by two unitary matrix to implement as a control operation
    Returns:
        circuit (QuantumCircuit): quantum circuit of Pn(U) gate
    """
    assert n_controls > 0, "number of controls must be 1 or more!"

    # target = n_controls[-1] +1 for now!
    circuit = QuantumCircuit(n_controls + 1)

    for k in reversed(range(2, n_controls + 1)):
        power = n_controls - k + 1
        root_u = fractional_matrix_power(single_q_unitary, 1 / 2 ** (power))

        root_u_circ = QuantumCircuit(1)
        root_u_circ.unitary(root_u, 0)
        root_u_circ.name = f"U^1/{2 ** (power)}"
        root_u_circ_gate = root_u_circ.to_gate().control(1)

        circuit.append(root_u_circ_gate, [k - 1, n_controls])

    return circuit


def qn_gate(n_controls: int) -> QuantumCircuit:
    """
    Qn gate defined in equation 5 of https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        n_controls (int): number of controls
    Returns:
        circuit (QuantumCircuit): quantum circuit of Qn gate
    """
    assert n_controls > 0, "number of controls must be 1 or more!"

    if n_controls == 2:
        circuit = QuantumCircuit(n_controls + 1)
        circuit.crx(pi, 0, 1)
        return circuit

    circuit = QuantumCircuit(n_controls + 1)

    for j in reversed(range(2, n_controls)):
        circuit = circuit.compose(pn_gate(j))
        circuit.crx(pi / (2 ** (j - 1)), 0, j)

    # Q2 gate in paper
    circuit.crx(pi, 0, 1)
    for k in range(2, n_controls):
        circuit = circuit.compose(pn_gate(k).inverse())

    return circuit
