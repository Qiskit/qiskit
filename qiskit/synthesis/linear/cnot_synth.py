# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
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


def synth_cnot_count_full_pmh(
    state: list[list[bool]] | np.ndarray[bool], section_size: int = 2
) -> QuantumCircuit:
    """Synthesize linear reversible circuits using the Patel-Markov-Hayes algorithm with virtual padding."""

    if not isinstance(state, (list, np.ndarray)):
        raise QiskitError(
            "state should be of type list or numpy.ndarray, "
            "but was of the type {}".format(type(state))
        )
    state = np.array(state)
    num_qubits = state.shape[0]

    # Virtual Padding
    if section_size > num_qubits:
        padding_size = section_size - (num_qubits % section_size)
        state = np.hstack((state, np.zeros((num_qubits, padding_size), dtype=bool)))

    # Synthesize lower triangular part
    [state, circuit_l] = _lwr_cnot_synth(state, section_size, num_qubits)
    state = np.transpose(state)

    # Synthesize upper triangular part
    [state, circuit_u] = _lwr_cnot_synth(state, section_size, num_qubits)
    circuit_l.reverse()
    for i in circuit_u:
        i.reverse()

    # Convert the list into a circuit of C-NOT gates (removing virtual padding)
    circ = QuantumCircuit(num_qubits)  # Circuit size is the original num_qubits
    for i in circuit_u + circuit_l:
        # Only add gates if both control and target are within the original matrix
        if i[0] < num_qubits and i[1] < num_qubits:
            circ.cx(i[0], i[1])

    return circ

def _lwr_cnot_synth(state, section_size, num_qubits):
    """Helper function for the Patel-Markov-Hayes algorithm with virtual padding."""

    circuit = []
    cutoff = 1

    # Iterate over column sections (including padded sections)
    for sec in range(1, int(np.ceil(state.shape[1] / section_size)) + 1):
        # Remove duplicate sub-rows in section sec
        patt = {}
        for row in range((sec - 1) * section_size, num_qubits):
            sub_row_patt = copy.deepcopy(
                state[row, (sec - 1) * section_size : sec * section_size]
            )
            if np.sum(sub_row_patt) == 0:
                continue
            if str(sub_row_patt) not in patt:
                patt[str(sub_row_patt)] = row
            else:
                state[row, :] ^= state[patt[str(sub_row_patt)], :]
                circuit.append([patt[str(sub_row_patt)], row])

        # Use gaussian elimination for remaining entries in column section
        # Modified loop range
        for col in range((sec - 1) * section_size, min(sec * section_size, num_qubits)):
            # Check if 1 on diagonal
            diag_one = 1
            if state[col, col] == 0:
                diag_one = 0
            # Remove ones in rows below column col
            for row in range(col + 1, num_qubits):
                if state[row, col] == 1:
                    if diag_one == 0:
                        state[col, :] ^= state[row, :]
                        circuit.append([row, col])
                        diag_one = 1
                    state[row, :] ^= state[col, :]
                    circuit.append([col, row])
                # Back reduce the pivot row using the current row
                if sum(state[col, :] & state[row, :]) > cutoff:
                    state[col, :] ^= state[row, :]
                    circuit.append([row, col])

    return [state, circuit]
