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

from qiskit._accelerate.synthesis import synth_cnot_count_full_pmh as _fast


def synth_cnot_count_full_pmh(
    state: list[list[bool]] | np.ndarray[bool], section_size: int = 2
) -> QuantumCircuit:
    # normalize input
    print("input:\n", state)
    inp = np.asarray(state).astype(bool)
    lower_cnots, upper_cnots = _fast(inp, section_size)
    print("lower:", lower_cnots)
    print("upper:", upper_cnots)
    lower_cnots.reverse()
    upper_cnots = [(b, a) for a, b in upper_cnots]

    circ = QuantumCircuit(inp.shape[0])
    for i in upper_cnots + lower_cnots:
        circ.cx(i[0], i[1])
    return circ
