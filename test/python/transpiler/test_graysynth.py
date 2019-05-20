# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Gray-Synth algorithm"""

import numpy as np
from qiskit import BasicAer, execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.passes import graysynth
from qiskit.test import QiskitTestCase


class TestGraySynth(QiskitTestCase):
    """Test the Gray-Synth algorithm."""

    def test_gray_synth(self):
        """Test the Gray-Synth algorithm.

        The algorithm should take the following matrix as an input:
        S =
        [[0, 1, 1, 0, 1, 1],
         [0, 1, 1, 0, 1, 0],
         [0, 0, 0, 1, 1, 0],
         [1, 0, 0, 1, 1, 1],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0]]

        And should return the following circuit (or an equivalent one):
                          ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
        q_0: |0>──────────┤ X ├┤ X ├┤ T ├┤ X ├┤ X ├┤ X ├┤ X ├┤ T ├┤ X ├┤ T ├┤ X ├┤ X ├┤ T ├┤ X ├
                          └─┬─┘└─┬─┘└───┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘└───┘└─┬─┘└───┘└─┬─┘└─┬─┘└───┘└─┬─┘
        q_1: |0>────────────┼────┼─────────■────┼────┼────┼─────────┼─────────┼────┼─────────■──
                            │    │              │    │    │         │         │    │
        q_2: |0>───────■────■────┼──────────────■────┼────┼─────────┼────■────┼────┼────────────
                ┌───┐┌─┴─┐┌───┐  │                   │    │         │  ┌─┴─┐  │    │
        q_3: |0>┤ T ├┤ X ├┤ T ├──■───────────────────┼────┼─────────■──┤ X ├──┼────┼────────────
                └───┘└───┘└───┘                      │    │            └───┘  │    │
        q_4: |0>─────────────────────────────────────■────┼───────────────────■────┼────────────
                                                          │                        │
        q_5: |0>──────────────────────────────────────────■────────────────────────■────────────

        """

        n = 6
        m = 2
        cnots = [[0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                 [1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 0]]
        cnots = list(map(list, zip(*cnots)))
        c_gray = graysynth(cnots, n, m)  # Run the graysynth algorithm with S2
        backend_sim = BasicAer.get_backend('unitary_simulator')
        result_gray = execute(c_gray, backend_sim).result()
        unitary_gray = result_gray.get_unitary(c_gray)

        # Create the circuit displayed above:
        q = QuantumRegister(n, 'q')  # Create a Quantum Register with n qubits.
        c_compare = QuantumCircuit(q)  # Create a Quantum Circuit acting on the q register
        c_compare.t(q[3])
        c_compare.cx(q[2], q[3])
        c_compare.t(q[3])
        c_compare.cx(q[2], q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[2], q[0])
        c_compare.cx(q[4], q[0])
        c_compare.cx(q[5], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[3])
        c_compare.cx(q[4], q[0])
        c_compare.cx(q[5], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[1], q[0])
        result_compare = execute(c_compare, backend_sim).result()
        unitary_compare = result_compare.get_unitary(c_compare)

        # Check if the two circuits are equivalent
        self.assertTrue(np.allclose(unitary_gray, unitary_compare))
