# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=unused-wildcard-import, wildcard-import

"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""


# Redirect getattrs to modules new location
# TODO: Deprecate in 0.24.0 and remove in 0.26.0
from qiskit.synthesis.linear.graysynth import *


def cnot_synth(state, section_size=2):
    """
    Synthesize linear reversible circuits for all-to-all architecture
    using Patel, Markov and Hayes method.

    This function is an implementation of the Patel, Markov and Hayes algorithm from [1]
    for optimal synthesis of linear reversible circuits for all-to-all architecture,
    as specified by an n x n matrix.

    Args:
        state (list[list] or ndarray): n x n boolean invertible matrix, describing the state
            of the input circuit
        section_size (int): the size of each section, used in the
            Patel–Markov–Hayes algorithm [1]. section_size must be a factor of num_qubits.

    Returns:
        QuantumCircuit: a CX-only circuit implementing the linear transformation.

    Raises:
        QiskitError: when variable "state" isn't of type numpy.ndarray

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """
    return synth_cnot_count_full_pmh(state, section_size=section_size)
