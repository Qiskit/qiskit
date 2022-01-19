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
"""A module for calculating fidelity of an ansatz evaluated with provided parameter values and
a maximally entangled state of a compatible size."""
from typing import Union

from qiskit import QuantumCircuit
from qiskit.opflow import OperatorBase
from qiskit.providers import BaseBackend
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.utils import QuantumInstance


def calc_ansatz_mes_fidelity(
    ansatz_n_mes: OperatorBase, backend: Union[BaseBackend, QuantumInstance], num_of_mes: int
) -> float:
    """Calculates fidelity between n exact Maximally Entangled States (MES) and bound ansatz."""
    exact_n_mes = _build_n_mes(num_of_mes, backend)
    return state_fidelity(exact_n_mes, ansatz_n_mes)


def _build_n_mes(num_states: int, backend: Union[BaseBackend, QuantumInstance]) -> Statevector:
    """Builds n Maximally Entangled States (MES) as state vectors exactly."""
    qc = _build_mes()
    for _ in range(num_states - 1):
        qc = qc.tensor(_build_mes())

    return backend.run(qc).result().get_statevector()


def _build_mes() -> QuantumCircuit:
    """Builds a quantum circuit for a single Maximally Entangled State (MES)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    return qc
