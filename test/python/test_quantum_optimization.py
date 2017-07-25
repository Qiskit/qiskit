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
import sys
sys.path.append("../..")
from tools.optimizationtools import trial_circuit_ry

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

if __name__ == '__main__':
    unittest.main()
        
