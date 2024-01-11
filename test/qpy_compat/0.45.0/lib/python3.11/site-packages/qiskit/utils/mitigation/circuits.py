# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This code was originally copied from the qiskit-ignis repsoitory see:
# https://github.com/Qiskit/qiskit-ignis/blob/b91066c72171bcd55a70e6e8993b813ec763cf41/qiskit/ignis/mitigation/measurement/circuits.py
# it was migrated to qiskit-terra as qiskit-ignis is being deprecated

"""
Measurement calibration circuits. To apply the measurement mitigation
use the fitters to produce a filter.
"""
from typing import List, Tuple, Union
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    since="0.24.0",
    package_name="qiskit-terra",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def count_keys(num_qubits: int) -> List[str]:
    """Deprecated: Return ordered count keys.

    Args:
        num_qubits: The number of qubits in the generated list.
    Returns:
        The strings of all 0/1 combinations of the given number of qubits
    Example:
        >>> count_keys(3)
        ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [bin(j)[2:].zfill(num_qubits) for j in range(2**num_qubits)]


@deprecate_func(
    since="0.24.0",
    package_name="qiskit-terra",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def complete_meas_cal(
    qubit_list: List[int] = None,
    qr: Union[int, List["QuantumRegister"]] = None,
    cr: Union[int, List["ClassicalRegister"]] = None,
    circlabel: str = "",
) -> Tuple[List["QuantumCircuit"], List[str]]:
    """
    Deprecated: Return a list of measurement calibration circuits for the full
    Hilbert space.

    If the circuit contains :math:`n` qubits, then :math:`2^n` calibration circuits
    are created, each of which creates a basis state.

    Args:
        qubit_list: A list of qubits to perform the measurement correction on.
           If `None`, and qr is given then assumed to be performed over the entire
           qr. The calibration states will be labelled according to this ordering (default `None`).

        qr: Quantum registers (or their size).
            If ``None``, one is created (default ``None``).

        cr: Classical registers (or their size).
            If ``None``, one is created(default ``None``).

        circlabel: A string to add to the front of circuit names for
            unique identification(default ' ').

    Returns:
        A list of QuantumCircuit objects containing the calibration circuits.

        A list of calibration state labels.

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_1001.

        Pass the results of these circuits to the CompleteMeasurementFitter
        constructor.

    Raises:
        QiskitError: if both `qubit_list` and `qr` are `None`.

    """
    # Runtime imports to avoid circular imports causeed by QuantumInstance
    # getting initialized by imported utils/__init__ which is imported
    # by qiskit.circuit
    from qiskit.circuit.quantumregister import QuantumRegister
    from qiskit.circuit.classicalregister import ClassicalRegister
    from qiskit.circuit.exceptions import QiskitError

    if qubit_list is None and qr is None:
        raise QiskitError("Must give one of a qubit_list or a qr")

    # Create the registers if not already done
    if qr is None:
        qr = QuantumRegister(max(qubit_list) + 1)

    if isinstance(qr, int):
        qr = QuantumRegister(qr)

    if qubit_list is None:
        qubit_list = range(len(qr))

    if isinstance(cr, int):
        cr = ClassicalRegister(cr)

    nqubits = len(qubit_list)

    # labels for 2**n qubit states
    state_labels = count_keys(nqubits)

    cal_circuits, _ = tensored_meas_cal([qubit_list], qr, cr, circlabel)

    return cal_circuits, state_labels


@deprecate_func(
    since="0.24.0",
    package_name="qiskit-terra",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def tensored_meas_cal(
    mit_pattern: List[List[int]] = None,
    qr: Union[int, List["QuantumRegister"]] = None,
    cr: Union[int, List["ClassicalRegister"]] = None,
    circlabel: str = "",
) -> Tuple[List["QuantumCircuit"], List[List[int]]]:
    """
    Deprecated: Return a list of calibration circuits

    Args:
        mit_pattern: Qubits on which to perform the
            measurement correction, divided to groups according to tensors.
            If `None` and `qr` is given then assumed to be performed over the entire
            `qr` as one group (default `None`).

        qr: A quantum register (or its size).
        If `None`, one is created (default `None`).

        cr: A classical register (or its size).
        If `None`, one is created (default `None`).

        circlabel: A string to add to the front of circuit names for
            unique identification (default ' ').

    Returns:
        A list of two QuantumCircuit objects containing the calibration circuits
        mit_pattern

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_000 and cal_111.

        Pass the results of these circuits to the TensoredMeasurementFitter
        constructor.

    Raises:
        QiskitError: if both `mit_pattern` and `qr` are None.
        QiskitError: if a qubit appears more than once in `mit_pattern`.

    """
    # Runtime imports to avoid circular imports causeed by QuantumInstance
    # getting initialized by imported utils/__init__ which is imported
    # by qiskit.circuit
    from qiskit.circuit.quantumregister import QuantumRegister
    from qiskit.circuit.classicalregister import ClassicalRegister
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from qiskit.circuit.exceptions import QiskitError

    if mit_pattern is None and qr is None:
        raise QiskitError("Must give one of mit_pattern or qr")

    if isinstance(qr, int):
        qr = QuantumRegister(qr)

    qubits_in_pattern = []
    if mit_pattern is not None:
        for qubit_list in mit_pattern:
            for qubit in qubit_list:
                if qubit in qubits_in_pattern:
                    raise QiskitError(
                        "mit_pattern cannot contain multiple instances of the same qubit"
                    )
                qubits_in_pattern.append(qubit)

        # Create the registers if not already done
        if qr is None:
            qr = QuantumRegister(max(qubits_in_pattern) + 1)
    else:
        qubits_in_pattern = range(len(qr))
        mit_pattern = [qubits_in_pattern]

    nqubits = len(qubits_in_pattern)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    if isinstance(cr, int):
        cr = ClassicalRegister(cr)

    qubits_list_sizes = [len(qubit_list) for qubit_list in mit_pattern]
    nqubits = sum(qubits_list_sizes)
    size_of_largest_group = max(qubits_list_sizes)
    largest_labels = count_keys(size_of_largest_group)

    state_labels = []
    for largest_state in largest_labels:
        basis_state = ""
        for list_size in qubits_list_sizes:
            basis_state = largest_state[:list_size] + basis_state
        state_labels.append(basis_state)

    cal_circuits = []
    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(qr, cr, name=f"{circlabel}cal_{basis_state}")

        end_index = nqubits
        for qubit_list, list_size in zip(mit_pattern, qubits_list_sizes):

            start_index = end_index - list_size
            substate = basis_state[start_index:end_index]

            for qind in range(list_size):
                if substate[list_size - qind - 1] == "1":
                    qc_circuit.x(qr[qubit_list[qind]])

            end_index = start_index

        qc_circuit.barrier(qr)

        # add measurements
        end_index = nqubits
        for qubit_list, list_size in zip(mit_pattern, qubits_list_sizes):

            for qind in range(list_size):
                qc_circuit.measure(qr[qubit_list[qind]], cr[nqubits - (end_index - qind)])

            end_index -= list_size

        cal_circuits.append(qc_circuit)

    return cal_circuits, mit_pattern
