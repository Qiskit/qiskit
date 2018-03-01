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

import sys
import os
import unittest

from qiskit import QuantumProgram
from qiskit import qasm, unroll
from .common import QiskitTestCase


class UnrollerTest(QiskitTestCase):
    """Test the unroller."""

    def setUp(self):
        self.seed = 42
        self.qp = QuantumProgram()

    def test_dagunroller(self):
        ast = qasm.Qasm(filename=self._get_resource_path('qasm/example.qasm')).parse()
        basis = ["cx", "u1", "u2", "u3"]
        unr = unroll.Unroller(ast, unroll.DAGBackend(basis))
        dag_circuit = unr.execute()

        dag_unroller = unroll.DagUnroller(dag_circuit, unroll.DAGBackend(basis))
        unrolled_dag_circuit = dag_unroller.execute()
        expanded_dag_circuit = dag_unroller.expand_gates()

        self.assertEqual( unrolled_dag_circuit, expanded_dag_circuit)

