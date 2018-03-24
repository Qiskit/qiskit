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
"""Tests for visualization tools."""

import os
import random
from inspect import signature
import unittest

from qiskit import QuantumProgram
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from .common import QiskitTestCase

try:
    from qiskit.tools.visualization import latex_drawer
    from qiskit.tools.visualization import circuit_drawer
    VALID_MATPLOTLIB = True
except RuntimeError:
    # Under some combinations (travis osx vms, or headless configurations)
    # matplotlib might not be fully, raising:
    # RuntimeError: Python is not installed as a framework.
    # when importing. If that is the case, the full test is skipped.
    VALID_MATPLOTLIB = False


@unittest.skipIf(not VALID_MATPLOTLIB, 'osx matplotlib backend not avaiable')
class TestLatexDrawer(QiskitTestCase):
    """QISKit latex drawer tests."""

    def setUp(self):
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit('latex_test', [qr], [cr])
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[1], cr[1])
        qc.x(qr[1]).c_if(cr, 1)
        qc.measure(qr, cr)
        self.qp = qp
        self.qc = qc
        self.qobj = qp.compile(['latex_test'])

    def test_latex_drawer(self):
        filename = self._get_resource_path('test_latex_drawer.tex')
        try:
            latex_drawer(self.qc, filename)
        except Exception:
            if os.path.exists(filename):
                os.remove(filename)
            raise

    def test_teleport(self):
        filename = self._get_resource_path('test_teleport.tex')
        QPS_SPECS = {
            "circuits": [{
                "name": "teleport",
                "quantum_registers": [{
                    "name": "q",
                    "size": 3
                }],
                "classical_registers": [
                    {"name": "c0",
                     "size": 1},
                    {"name": "c1",
                     "size": 1},
                    {"name": "c2",
                     "size": 1},
                ]}]
        }

        qp = QuantumProgram(specs=QPS_SPECS)
        qc = qp.get_circuit("teleport")
        q = qp.get_quantum_register("q")
        c0 = qp.get_classical_register("c0")
        c1 = qp.get_classical_register("c1")
        c2 = qp.get_classical_register("c2")

        # Prepare an initial state
        qc.u3(0.3, 0.2, 0.1, q[0])

        # Prepare a Bell pair
        qc.h(q[1])
        qc.cx(q[1], q[2])

        # Barrier following state preparation
        qc.barrier(q)

        # Measure in the Bell basis
        qc.cx(q[0], q[1])
        qc.h(q[0])
        qc.measure(q[0], c0[0])
        qc.measure(q[1], c1[0])

        # Apply a correction
        qc.z(q[2]).c_if(c0, 1)
        qc.x(q[2]).c_if(c1, 1)
        qc.measure(q[2], c2[0])
        try:
            latex_drawer(qc, filename)
        except Exception:
            if os.path.exists(filename):
                os.remove(filename)
            raise


@unittest.skipIf(not VALID_MATPLOTLIB, 'osx matplotlib backend not avaiable')
class TestCircuitDrawer(QiskitTestCase):
    """QISKit circuit drawer tests."""

    def randomCircuit(self, width, depth, max_operands):
        """Generate random circuit of arbitrary size.
        Note: the depth is the layers of independent operation. true depth
        in the image may be more for visualization purposes, if gates overlap.

        Args:
            width (int): number of quantum wires
            depth (int): layers of operations
            max_operands (int): maximum operands of each gate

        Returns:
            QuantumCircuit: constructed circuit
        """
        width = 3
        depth = 3
        max_operands = 3

        qr = QuantumRegister("q", width)
        qc = QuantumCircuit(qr)

        one_q_ops = "iden,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz"
        two_q_ops = "cx,cy,cz,ch,crz,cu1,cu3,swap"
        three_q_ops = "ccx"

        # apply arbitrary random operations at every depth
        for _ in range(depth):
            # choose either 1, 2, or 3 qubits for the operation
            remaining_qubits = list(range(width))
            while remaining_qubits:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = random.choice(range(max_possible_operands))+1
                operands = random.sample(remaining_qubits, num_operands)
                remaining_qubits = [q for q in remaining_qubits if q not in operands]
                if num_operands == 1:
                    op = random.choice(one_q_ops.split(','))
                elif num_operands == 2:
                    op = random.choice(two_q_ops.split(','))
                elif num_operands == 3:
                    op = random.choice(three_q_ops.split(','))
                # every gate is defined as a method of the QuantumCircuit class
                # the code below is so we can call a gate by its name
                gate = getattr(QuantumCircuit, op)
                op_args = list(signature(gate).parameters.keys())
                num_angles = len(op_args) - num_operands - 1    # -1 for the 'self' arg
                angles = [random.uniform(0, 3.14) for x in range(num_angles)]
                register_operands = [qr[i] for i in operands]
                gate(qc, *angles, *register_operands)

        return qc

    def test_tiny_circuit(self):
        qc = self.randomCircuit(1, 1, 1)
        im = circuit_drawer(qc)
        self.assertNotEqual(im, None)

    def test_normal_circuit(self):
        qc = self.randomCircuit(5, 5, 3)
        im = circuit_drawer(qc)
        self.assertNotEqual(im, None)

    def test_wide_circuit(self):
        qc = self.randomCircuit(100, 1, 1)
        im = circuit_drawer(qc)
        self.assertNotEqual(im, None)

    def test_deep_circuit(self):
        qc = self.randomCircuit(1, 100, 1)
        im = circuit_drawer(qc)
        self.assertNotEqual(im, None)

    def test_huge_circuit(self):
        qc = self.randomCircuit(40, 15, 1)
        im = circuit_drawer(qc)
        self.assertNotEqual(im, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
