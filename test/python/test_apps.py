# -*- coding: utf-8 -*-

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
import sys
import numpy as np
import unittest
import logging
import os
from scipy import linalg as la
sys.path.append("../..")
from tools.apps.optimization import trial_circuit_ry
from tools.apps.optimization import Energy_Estimate, make_Hamiltonian, Hamiltonian_from_file
from tools.qi.pauli import Pauli

class TestQuantumOptimization(unittest.TestCase):
    """Tests for quantum optimization"""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.logFileName = cls.moduleName + '.log'
        log_fmt = 'TestQuantumOptimization:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)


    def test_trial_functions(self):
        entangler_map = {0: [2], 1: [2], 3: [2], 4: [2]}

        m = 1
        n = 6
        theta = np.zeros(m * n)

        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map)

        logging.info(trial_circuit.qasm())

        logging.info("With No measurement:\n")
        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, None, None)

        logging.info(trial_circuit.qasm())

        logging.info("With Y measurement:\n")
        meas_sting = ['Y' for x in range(n)]

        trial_circuit = trial_circuit_ry(n, m, theta, entangler_map, meas_sting)

        logging.info(trial_circuit.qasm())


class TestHamiltonian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.logFileName = cls.moduleName + '.log'
        log_fmt = 'TestHamiltonian:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)

    def test_hamiltonian(self):
        # printing an example from a H2 file
        hfile = os.path.dirname(__file__) + "/H2Equilibrium.txt"
        logging.info(make_Hamiltonian(Hamiltonian_from_file(hfile)))

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

        pauli_list = [(1, Pauli(v0, np.zeros(n))), (1, Pauli(v1, np.zeros(n))), (1, Pauli(v2, np.zeros(n))), (1, Pauli(v3, np.zeros(n)))]
        a = make_Hamiltonian(pauli_list)
        logging.info(a)

        w, v = la.eigh(a, eigvals=(0, 0))
        logging.info(w)
        logging.info(v)

        data = {'000': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'001': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'010': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'011': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'100': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'101': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'110': 10}
        logging.info(Energy_Estimate(data, pauli_list))
        data = {'111': 10}
        logging.info(Energy_Estimate(data, pauli_list))

if __name__ == '__main__':
    unittest.main()
