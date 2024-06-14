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
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""

from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError

from qiskit._accelerate.synthesis import synth_cnot_count_full_pmh as fast_pmh


def synth_cnot_count_full_pmh(
    state: list[list[bool]] | np.ndarray[bool], section_size: int = 2
) -> QuantumCircuit:
    # call Rust implementation with normalized input
    circuit_data = fast_pmh(np.asarray(state).astype(bool), section_size)

    # construct circuit from the data
    return QuantumCircuit._from_circuit_data(circuit_data)
