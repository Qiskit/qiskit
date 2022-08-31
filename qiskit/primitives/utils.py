# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utility functions for primitives
"""

from __future__ import annotations

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli


def init_circuit(state: QuantumCircuit | Statevector) -> QuantumCircuit:
    """Initialize state by converting the input to a quantum circuit.

    Args:
        state: The state as quantum circuit or statevector.

    Returns:
        The state as quantum circuit.
    """
    if isinstance(state, QuantumCircuit):
        return state
    if not isinstance(state, Statevector):
        state = Statevector(state)
    qc = QuantumCircuit(state.num_qubits)
    qc.append(Initialize(state), qargs=range(state.num_qubits))
    return qc


def init_observable(observable: BaseOperator | PauliSumOp) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        observable: The observable.

    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.

    Raises:
        TypeError: If the observable is a :class:`~qiskit.opflow.PauliSumOp` and has a parameterized
            coefficient.
    """
    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, PauliSumOp):
        if isinstance(observable.coeff, ParameterExpression):
            raise TypeError(
                f"Observable must have numerical coefficient, not {type(observable.coeff)}."
            )
        return observable.coeff * observable.primitive
    elif isinstance(observable, BasePauli):
        return SparsePauliOp(observable)
    elif isinstance(observable, BaseOperator):
        return SparsePauliOp.from_operator(observable)
    else:
        return SparsePauliOp(observable)


def final_measurement_mapping(circuit: QuantumCircuit) -> dict[int, int]:
    """Return the final measurement mapping for the circuit.

    Dict keys label measured qubits, whereas the values indicate the
    classical bit onto which that qubits measurement result is stored.

    Note: this function is a slightly simplified version of a utility function
    ``_final_measurement_mapping`` of
    `mthree <https://github.com/Qiskit-Partners/mthree>`_.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of qubits to classical bits for final measurements.
    """
    active_qubits = list(range(circuit.num_qubits))
    active_cbits = list(range(circuit.num_clbits))

    # Find final measurements starting in back
    mapping = {}
    for item in circuit._data[::-1]:
        if item[0].name == "measure":
            cbit = circuit.find_bit(item[2][0]).index
            qbit = circuit.find_bit(item[1][0]).index
            if cbit in active_cbits and qbit in active_qubits:
                mapping[qbit] = cbit
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item[0].name != "barrier":
            for qq in item[1]:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    return mapping
