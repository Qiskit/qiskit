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

    N_qubits = max(*control_list, target_q) + 1
    circuit = QuantumCircuit(N_qubits)
    CnU_gate = _CnU(len(control_list), single_q_unitary).to_gate()
    circuit.append(CnU_gate, [*control_list, target_q])
    return circuit


def _CnU(n_controls: int, U_array: array) -> QuantumCircuit:
    """
    Implement a multicontrol U gate according to https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        n_controls(int): number of control qubits
        U_array (array): two by two unitary matrix to implement as a control operation
    Returns:
        circuit (QuantumCircuit): Quantum circuit implementing multicontrol U_array
    """
    assert U_array.shape == (2, 2), "input U_array is not a single qubit gate"
    assert allclose(U_array @ U_array.conj().T, eye(2)), "input U_array is not unitary"

    targ = n_controls + 1
    circuit = QuantumCircuit(targ)
    if n_controls == 1:
        U_circ = QuantumCircuit(1)
        U_circ.unitary(U_array, 0)
        U_circ.name = "U"
        U_circ_gate = U_circ.to_gate().control(1)
        circuit.append(U_circ_gate, [0, n_controls])
    else:
        PnU = _Pn_U_gate(n_controls, U_array)
        circuit = circuit.compose(PnU)

        power = n_controls - 1
        root_U = fractional_matrix_power(U_array, 1 / 2 ** (n_controls - 1))
        root_U_circ = QuantumCircuit(1)
        root_U_circ.unitary(root_U, 0)
        root_U_circ.name = f"U^1/{2 ** power}"
        control_root_U_gate = root_U_circ.to_gate().control(1)
        circuit.append(control_root_U_gate, [0, n_controls])

        Qn = _Qn_gate(n_controls)
        circuit = circuit.compose(Qn)
        circuit = circuit.compose(PnU.inverse())
        circuit = circuit.compose(Qn.inverse())

    return circuit


def _Pn_gate(n_controls: int) -> QuantumCircuit:
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


def _Pn_U_gate(n_controls: int, U_gate: array) -> QuantumCircuit:
    """
    Pn(U) gate defined in equation 2 of https://doi.org/10.1103/PhysRevA.106.042602

    Args:
        n_controls (int): number of controls
    Returns:
        circuit (QuantumCircuit): quantum circuit of Pn(U) gate
    """
    assert n_controls > 0, "number of controls must be 1 or more!"

    # target = n_controls[-1] +1 for now!
    circuit = QuantumCircuit(n_controls + 1)

    for k in reversed(range(2, n_controls + 1)):
        power = n_controls - k + 1
        root_U = fractional_matrix_power(U_gate, 1 / 2 ** (power))

        root_U_circ = QuantumCircuit(1)
        root_U_circ.unitary(root_U, 0)
        root_U_circ.name = f"U^1/{2 ** (power)}"
        control_root_U_gate = root_U_circ.to_gate().control(1)

        circuit.append(control_root_U_gate, [k - 1, n_controls])

    return circuit


def _Qn_gate(n_controls: int) -> QuantumCircuit:
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
        circuit = circuit.compose(_Pn_gate(j))
        circuit.crx(pi / (2 ** (j - 1)), 0, j)

    # Q2 gate in paper
    circuit.crx(pi, 0, 1)
    for k in range(2, n_controls):
        circuit = circuit.compose(_Pn_gate(k).inverse())

    return circuit
