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
from qiskit.synthesis.linear.cnot_synth import *
from qiskit.synthesis.linear_phase.cnot_phase_synth import *


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


def graysynth(cnots, angles, section_size=2):
    """This function is an implementation of the GraySynth algorithm of
    Amy, Azimadeh and Mosca.

    GraySynth is a heuristic algorithm from [1] for synthesizing small parity networks.
    It is inspired by Gray codes. Given a set of binary strings S
    (called "cnots" bellow), the algorithm synthesizes a parity network for S by
    repeatedly choosing an index i to expand and then effectively recursing on
    the co-factors S_0 and S_1, consisting of the strings y in S,
    with y_i = 0 or 1 respectively. As a subset S is recursively expanded,
    CNOT gates are applied so that a designated target bit contains the
    (partial) parity ksi_y(x) where y_i = 1 if and only if y'_i = 1 for all
    y' in S. If S is a singleton {y'}, then y = y', hence the target bit contains
    the value ksi_y'(x) as desired.

    Notably, rather than uncomputing this sequence of CNOT gates when a subset S
    is finished being synthesized, the algorithm maintains the invariant
    that the remaining parities to be computed are expressed over the current state
    of bits. This allows the algorithm to avoid the 'backtracking' inherent in
    uncomputing-based methods.

    The algorithm is described in detail in section 4 of [1].

    Args:
        cnots (list[list]): a matrix whose columns are the parities to be synthesized
            e.g.::

                [[0, 1, 1, 1, 1, 1],
                 [1, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 1, 0]]

            corresponds to::

                 x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1

        angles (list): a list containing all the phase-shift gates which are
            to be applied, in the same order as in "cnots". A number is
            interpreted as the angle of p(angle), otherwise the elements
            have to be 't', 'tdg', 's', 'sdg' or 'z'.

        section_size (int): the size of every section, used in _lwr_cnot_synth(), in the
            Patel–Markov–Hayes algorithm. section_size must be a factor of num_qubits.

    Returns:
        QuantumCircuit: the decomposed quantum circuit.

    Raises:
        QiskitError: when dimensions of cnots and angles don't align.

    References:
        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.
           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,
           Quantum Science and Technology 4.1 (2018): 015002.
           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
    """
    return synth_cnot_phase_aam(cnots, angles, section_size=section_size)
