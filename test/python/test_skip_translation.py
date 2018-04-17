# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

from sys import version_info
import cProfile
import io
import pstats
import shutil
import time
import unittest

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from qiskit import qasm, unroll, QuantumProgram, QuantumJob
from qiskit.backends.local.qasmsimulator import QasmSimulator

from ._random_qasm_generator import RandomQasmGenerator
from .common import QiskitTestCase


class CompileSkipTranslationTest(QiskitTestCase):
    """Test compilaton with skip translation."""

    def setUp(self):
        self.seed = 88
        self.qasm_filename = self._get_resource_path('qasm/example.qasm')
        self.qp = QuantumProgram()
        self.qp.load_qasm_file(self.qasm_filename, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm('example')).parse(),
            unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        circuit_config = {'coupling_map': None,
                          'basis_gates': 'u1,u2,u3,cx,id',
                          'layout': None,
                          'seed': self.seed}
        resources = {'max_credits': 3,
                     'wait': 5,
                     'timeout': 120}
        self.qobj = {'id': 'test_sim_single_shot',
                     'config': {
                         'max_credits': resources['max_credits'],
                         'shots': 1024,
                         'backend_name': 'local_qasm_simulator',
                     },
                     'circuits': [
                         {
                             'name': 'test',
                             'compiled_circuit': circuit,
                             'compiled_circuit_qasm': None,
                             'config': circuit_config
                         }
                     ]}
        self.q_job = QuantumJob(self.qobj,
                                backend=QasmSimulator(),
                                circuit_config=circuit_config,
                                seed=self.seed,
                                resources=resources,
                                preformatted=True)

    def tearDown(self):
        pass

    def test_qasm_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        self.qobj['config']['shots'] = shots
        result = QasmSimulator().run(self.q_job)
        self.assertEqual(result.get_status(), 'COMPLETED')

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = QasmSimulator().run(self.q_job)
        shots = 1024
        threshold = 0.025 * shots
        counts = result.get_counts('test')
        target = {'100 100': shots / 8, '011 011': shots / 8,
                  '101 101': shots / 8, '111 111': shots / 8,
                  '000 000': shots / 8, '010 010': shots / 8,
                  '110 110': shots / 8, '001 001': shots / 8}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_if_statement(self):
        self.log.info('test_if_statement_x')
        shots = 100
        max_qubits = 3
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', max_qubits)
        cr = qp.create_classical_register('cr', max_qubits)
        circuit_if_true = qp.create_circuit('test_if_true', [qr], [cr])
        circuit_if_true.x(qr[0])
        circuit_if_true.x(qr[1])
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.x(qr[2]).c_if(cr, 0x3)
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.measure(qr[2], cr[2])

        result = qp.compile(['test_if_true'], backend='local_qasm_simulator', shots=1024,
                            skip_translation=True)

        config = {'max_credits': 10, 'shots': 1024, 'backend_name': 'local_qasm_simulator'}
        circuits = [{'name': 'test_if_true',
                     'config': {'coupling_map': None, 'basis_gates': 'u1,u2,u3,cx,id',
                                'seed': None, 'layout': None},
                     'compiled_circuit_qasm': 'OPENQASM 2.0;\n'
                                              'include "qelib1.inc";\n'
                                              'qreg qr[3];\n'
                                              'creg cr[3];\n'
                                              'x qr[0];\n'
                                              'x qr[1];\n'
                                              'measure qr[0] -> cr[0];\n'
                                              'measure qr[1] -> cr[1];\n'
                                              'if(cr==3) x qr[2];\n'
                                              'measure qr[0] -> cr[0];\n'
                                              'measure qr[1] -> cr[1];\n'
                                              'measure qr[2] -> cr[2];\n',
                     'compiled_circuit': {'operations': [
                         {'name': 'x', 'params': [], 'texparams': [], 'qubits': [1]},
                         {'name': 'measure', 'qubits': [1], 'clbits': [1]},
                         {'name': 'x', 'params': [], 'texparams': [], 'qubits': [0]},
                         {'name': 'measure', 'qubits': [0], 'clbits': [0]},
                         {'name': 'x', 'params': [], 'texparams': [], 'qubits': [2],
                          'conditional': {'type': 'equals', 'mask': '0x7', 'val': '0x3'}},
                         {'name': 'measure', 'qubits': [2], 'clbits': [2]},
                         {'name': 'measure', 'qubits': [1], 'clbits': [1]},
                         {'name': 'measure', 'qubits': [0], 'clbits': [0]}],
                         'header': {'number_of_qubits': 3, 'number_of_clbits': 3,
                                    'qubit_labels': [['qr', 0], ['qr', 1],
                                                     ['qr', 2]],
                                    'clbit_labels': [['cr', 3]]}}}]
        self.assertEqual(result['config'], config)
        self.assertEqual(result['circuits'], circuits)

    def test_teleport(self):
        """test teleportation as in tutorials"""

        self.log.info('test_teleport')
        pi = np.pi
        shots = 1000
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 3)
        cr0 = qp.create_classical_register('cr0', 1)
        cr1 = qp.create_classical_register('cr1', 1)
        cr2 = qp.create_classical_register('cr2', 1)
        circuit = qp.create_circuit('teleport', [qr],
                                    [cr0, cr1, cr2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.ry(pi / 4, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr0[0])
        circuit.measure(qr[1], cr1[0])
        circuit.z(qr[2]).c_if(cr0, 1)
        circuit.x(qr[2]).c_if(cr1, 1)
        circuit.measure(qr[2], cr2[0])
        backend = 'local_qasm_simulator'
        result = qp.compile('teleport', backend=backend, shots=shots,
                            seed=self.seed, skip_translation=True)
        config = {'max_credits': 10, 'shots': 1000, 'backend_name': 'local_qasm_simulator'}
        circuits = [{'name': 'teleport',
                     'config': {'coupling_map': None, 'basis_gates': 'u1,u2,u3,cx,id', 'seed': 88,
                                'layout': None},
                     'compiled_circuit_qasm': 'OPENQASM 2.0;\n'
                                              'include "qelib1.inc";\n'
                                              'qreg qr[3];\n'
                                              'creg cr0[1];\n'
                                              'creg cr1[1];\n'
                                              'creg cr2[1];\n'
                                              'h qr[1];\n'
                                              'cx qr[1],qr[2];\n'
                                              'ry(0.785398163397448) qr[0];\n'
                                              'cx qr[0],qr[1];\n'
                                              'h qr[0];\n'
                                              'barrier qr[0],qr[1],qr[2];\n'
                                              'measure qr[0] -> cr0[0];\n'
                                              'measure qr[1] -> cr1[0];\n'
                                              'if(cr0==1) z qr[2];\n'
                                              'if(cr1==1) x qr[2];\n'
                                              'measure qr[2] -> cr2[0];\n',
                     'compiled_circuit': {'operations': [
                         {'name': 'h', 'params': [], 'texparams': [], 'qubits': [1]},
                         {'name': 'cx', 'params': [], 'texparams': [], 'qubits': [1, 2]},
                         {'name': 'ry', 'params': [0.7853981633974483],
                          'texparams': ['0.785398163397448'], 'qubits': [0]},
                         {'name': 'cx', 'params': [], 'texparams': [], 'qubits': [0, 1]},
                         {'name': 'h', 'params': [], 'texparams': [], 'qubits': [0]},
                         {'name': 'barrier', 'qubits': [0, 1, 2]},
                         {'name': 'measure', 'qubits': [1], 'clbits': [1]},
                         {'name': 'measure', 'qubits': [0], 'clbits': [0]},
                         {'name': 'z', 'params': [], 'texparams': [], 'qubits': [2],
                          'conditional': {'type': 'equals', 'mask': '0x1', 'val': '0x1'}},
                         {'name': 'x', 'params': [], 'texparams': [], 'qubits': [2],
                          'conditional': {'type': 'equals', 'mask': '0x2', 'val': '0x1'}},
                         {'name': 'measure', 'qubits': [2], 'clbits': [2]}],
                         'header': {'number_of_qubits': 3, 'number_of_clbits': 3,
                                    'qubit_labels': [['qr', 0], ['qr', 1],
                                                     ['qr', 2]],
                                    'clbit_labels': [['cr0', 1], ['cr1', 1],
                                                     ['cr2', 1]]}}}]
        self.assertEqual(result['config'], config)
        self.assertEqual(result['circuits'], circuits)


if __name__ == '__main__':
    unittest.main()
