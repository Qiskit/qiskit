# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utility funtions for expectation value classes
"""

from dataclasses import dataclass
from typing import Union

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.extensions import Initialize
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


def init_circuit(state: Union[QuantumCircuit, Statevector]) -> QuantumCircuit:
    """Initialize state."""
    if isinstance(state, QuantumCircuit):
        return state
    if not isinstance(state, Statevector):
        state = Statevector(state)
    qc = QuantumCircuit(state.num_qubits)
    qc.append(Initialize(state), qargs=range(state.num_qubits))
    return qc


def init_observable(observable: Union[BaseOperator, PauliSumOp]) -> SparsePauliOp:
    """Initialize observable"""
    if isinstance(observable, SparsePauliOp):
        return observable
    if isinstance(observable, PauliSumOp):
        if isinstance(observable.coeff, ParameterExpression):
            raise TypeError(
                f"observable must have numerical coefficient, not {type(observable.coeff)}"
            )
        return observable.coeff * observable.primitive
    if isinstance(observable, BaseOperator):
        return SparsePauliOp.from_operator(observable)
    return SparsePauliOp(observable)


@dataclass(frozen=True)
class Group:
    """The dataclass represents indices of circuit and observable."""

    circuit_index: int
    observable_index: int
