# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Test the the trial functions."""

import unittest

from scipy import linalg as la
import numpy as np

from qiskit.tools.apps.optimization import (Energy_Estimate, make_Hamiltonian,
                                            Hamiltonian_from_file,
                                            trial_circuit_ry)
from qiskit.tools.qi.pauli import Pauli

from .common import QiskitTestCase


class TestQuantumOptimization(QiskitTestCase):
    """Tests for quantum optimization"""

    def test_trial_functions(self):
        entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

        m = 1
        n = 6
        theta = np.zeros(m * n)

        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 456)

        self.log.info("With No measurement:\n")
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, None, None)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 324)

        self.log.info("With Y measurement:\n")
        meas_sting = ['Y' for x in range(n)]
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, meas_sting)
        qasm_txt = trial_circuit.qasm()
        self.log.info(qasm_txt)
        self.assertEqual(len(qasm_txt), 564)


class TestHamiltonian(QiskitTestCase):
    def test_hamiltonian(self):
        # printing an example from a H2 file
        hfile = self._get_resource_path("H2Equilibrium.txt")
        hamiltonian = make_Hamiltonian(Hamiltonian_from_file(hfile))
        self.log.info(hamiltonian)
        # [[-0.24522469381221926 0 0 0.18093133934472627 ]
        # [0 -1.0636560168497590 0.18093133934472627 0]
        # [0 0.18093133934472627 -1.0636560168497592 0]
        # [0.18093133934472627 0 0 -1.8369675149908681]]
        self.assertSequenceEqual([str(i) for i in hamiltonian[0]],
                                 ['(-0.245224693812+0j)', '0j', '0j', '(0.180931339345+0j)'])
        self.assertSequenceEqual([str(i) for i in hamiltonian[1]],
                                 ['0j', '(-1.06365601685+0j)', '(0.180931339345+0j)', '0j'])
        self.assertSequenceEqual([str(i) for i in hamiltonian[2]],
                                 ['0j', '(0.180931339345+0j)', '(-1.06365601685+0j)', '0j'])
        self.assertSequenceEqual([str(i) for i in hamiltonian[3]],
                                 ['(0.180931339345+0j)', '0j', '0j', '(-1.83696751499+0j)'])

        # printing an example from a graph input
        n = 3
        v0 = np.zeros(n)
        v0[2] = 1
        v1 = np.zeros(n)
        v1[0] = 1
        v1[1] = 1
        v2 = np.zeros(n)
        v2[0] = 1
        v2[2] = 1
        v3 = np.zeros(n)
        v3[1] = 1
        v3[2] = 1

        pauli_list = [(1, Pauli(v0, np.zeros(n))), (1, Pauli(v1, np.zeros(n))),
                      (1, Pauli(v2, np.zeros(n))), (1, Pauli(v3, np.zeros(n)))]
        a = make_Hamiltonian(pauli_list)
        self.log.info(a)

        w, v = la.eigh(a, eigvals=(0, 0))
        self.log.info(w)
        self.log.info(v)

        data = {'000': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'001': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'010': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'011': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'100': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'101': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'110': 10}
        self.log.info(Energy_Estimate(data, pauli_list))
        data = {'111': 10}
        self.log.info(Energy_Estimate(data, pauli_list))


if __name__ == '__main__':
    unittest.main()
