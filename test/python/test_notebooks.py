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
"""Test Jupyter notebooks

This module tests all jupyter notebooks ending in ".ipynb" in the
tutorial/sections directory of this repository.
"""
import unittest
import logging
import os
import sys
import numpy as np
from qiskit import QuantumProgram
import qiskit.qasm as qasm
import qiskit.unroll as unroll
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob


class TestJupyterNotebooks(unittest.TestCase):
    """Test jupyter notebooks in this repo"""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.logFileName = cls.moduleName + '.log'
        log_fmt = 'TestJupyterNotebooks:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)

    def setUp(self):
        path = os.path.dirname(__file__) + '/../../tutorial/sections/'
        self.notebook_path = os.path.abspath(path)
        notebook_glob = os.path.join(self.notebook_path, '*[!nbconvert].ipynb')
        self.notebook_file_list = glob.glob(notebook_glob)
        self.ep = ExecutePreprocessor(timeout=15, kernel_name='python3')

    def test_notebooks(self):
        for notebook_file in self.notebook_file_list:
            with self.subTest(name=notebook_file):
                with open(notebook_file) as notebook:
                    nb = nbformat.read(notebook,
                                       as_version=nbformat.NO_CONVERT)
                    self.ep.preprocess(nb, {'metadata':
                                            {'path': self.notebook_path}})
                    
        
if __name__ == '__main__':
    unittest.main()
