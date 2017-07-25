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
"""Quick program to test the Pauli class."""
import numpy as np
from scipy import linalg as la
import unittest
import logging
import os
import sys
sys.path.append("../..")
from tools.pauli import Pauli, random_pauli, inverse_pauli, pauli_group, sgn_prod

class TestPauli(unittest.TestCase):
    """Tests for Pauli class"""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.logFileName = cls.moduleName + '.log'
        log_fmt = 'TestPauli:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)

    def test_pauli(self):

        v = np.zeros(3)
        w = np.zeros(3)
        v[0] = 1
        w[1] = 1
        v[2] = 1
        w[2] = 1

        p = Pauli(v, w)
        logging.info(p)
        logging.info("In label form:")
        logging.info(p.to_label())
        logging.info("In matrix form:")
        logging.info(p.to_matrix())


        q = random_pauli(2)
        logging.info(q)

        r = inverse_pauli(p)
        logging.info("In label form:")
        logging.info(r.to_label())

        logging.info("Group in tensor order:")
        grp = pauli_group(3, case=1)
        for j in grp:
            logging.info(j.to_label())

        logging.info("Group in weight order:")
        grp = pauli_group(3)
        for j in grp:
            logging.info(j.to_label())

        logging.info("sign product:")
        p1 = Pauli(np.array([0]), np.array([1]))
        p2 = Pauli(np.array([1]), np.array([1]))
        p3, sgn = sgn_prod(p1, p2)
        logging.info(p1.to_label())
        logging.info(p2.to_label())
        logging.info(p3.to_label())
        logging.info(sgn)

        logging.info("sign product reverse:")
        p3, sgn = sgn_prod(p2, p1)
        logging.info(p2.to_label())
        logging.info(p1.to_label())
        logging.info(p3.to_label())
        logging.info(sgn)

if __name__ == '__main__':
    unittest.main()
        
