# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,redefined-builtin
from sys import version_info
import unittest

import numpy as np
from qiskit.qasm import Qasm
from qiskit.unroll import Unroller, DAGBackend, DagUnroller, JsonBackend
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import compile
from qiskit.backends.aer.qasm_simulator_py import QasmSimulatorPy
from qiskit.qobj import Qobj, QobjHeader, QobjConfig, QobjExperiment

from ..common import QiskitTestCase, bin_to_hex_keys


class TestAerQasmSimulatorPy(QiskitTestCase):
    """Test Aer's qasm_simulator_py."""

    def setUp(self):
        self.seed = 88
        self.backend = QasmSimulatorPy()
        backend_basis = self.backend.configuration().basis_gates
        qasm_filename = self._get_resource_path('qasm/example.qasm')
        qasm_ast = Qasm(filename=qasm_filename).parse()
        qasm_dag = Unroller(qasm_ast, DAGBackend()).execute()
        qasm_dag = DagUnroller(qasm_dag, DAGBackend(backend_basis)).expand_gates()
        qasm_json = DagUnroller(qasm_dag, JsonBackend(qasm_dag.basis)).execute()
        compiled_circuit = QobjExperiment.from_dict(qasm_json)
        compiled_circuit.header.name = 'test'

        self.qobj = Qobj(
            qobj_id='test_sim_single_shot',
            config=QobjConfig(
                shots=1024, memory_slots=6,
                max_credits=3, seed=self.seed
            ),
            experiments=[compiled_circuit],
            header=QobjHeader(backend_name='qasm_simulator_py')
        )

    def test_qasm_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        self.qobj.config.shots = shots
        result = QasmSimulatorPy().run(self.qobj).result()
        self.assertEqual(result.success, True)

    def test_qasm_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = self.backend.run(self.qobj).result()
        shots = 1024
        threshold = 0.04 * shots
        counts = result.get_counts('test')
        target = {'100 100': shots / 8, '011 011': shots / 8,
                  '101 101': shots / 8, '111 111': shots / 8,
                  '000 000': shots / 8, '010 010': shots / 8,
                  '110 110': shots / 8, '001 001': shots / 8}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_if_statement(self):
        shots = 100
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')

        circuit_if_true = QuantumCircuit(qr, cr)
        circuit_if_true.x(qr[0])
        circuit_if_true.x(qr[1])
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.x(qr[2]).c_if(cr, 0x3)
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.measure(qr[2], cr[2])

        circuit_if_false = QuantumCircuit(qr, cr)
        circuit_if_false.x(qr[0])
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.x(qr[2]).c_if(cr, 0x3)
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.measure(qr[2], cr[2])
        qobj = compile([circuit_if_true, circuit_if_false],
                       backend=self.backend, shots=shots, seed=self.seed)

        result = self.backend.run(qobj).result()
        counts_if_true = result.get_counts(circuit_if_true)
        counts_if_false = result.get_counts(circuit_if_false)
        self.assertEqual(counts_if_true, bin_to_hex_keys({'111': 100}))
        self.assertEqual(counts_if_false, bin_to_hex_keys({'001': 100}))

    @unittest.skipIf(version_info.minor == 5,
                     "Due to gate ordering issues with Python 3.5 "
                     "we have to disable this test until fixed")
    def test_teleport(self):
        """test teleportation as in tutorials"""
        self.log.info('test_teleport')
        pi = np.pi
        shots = 1000
        qr = QuantumRegister(3, 'qr')
        cr0 = ClassicalRegister(1, 'cr0')
        cr1 = ClassicalRegister(1, 'cr1')
        cr2 = ClassicalRegister(1, 'cr2')
        circuit = QuantumCircuit(qr, cr0, cr1, cr2, name='teleport')
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.ry(pi/4, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr0[0])
        circuit.measure(qr[1], cr1[0])
        circuit.z(qr[2]).c_if(cr0, 1)
        circuit.x(qr[2]).c_if(cr1, 1)
        circuit.measure(qr[2], cr2[0])
        qobj = compile(circuit, backend=self.backend, shots=shots, seed=self.seed)
        results = self.backend.run(qobj).result()
        data = results.get_counts('teleport')
        alice = {
            '00': data['0x0'] + data['0x4'],
            '01': data['0x2'] + data['0x6'],
            '10': data['0x1'] + data['0x5'],
            '11': data['0x3'] + data['0x7']
        }
        bob = {
            '0': data['0x0'] + data['0x2'] + data['0x1'] + data['0x3'],
            '1': data['0x4'] + data['0x6'] + data['0x5'] + data['0x7']
        }
        self.log.info('test_teleport: circuit:')
        self.log.info('test_teleport: circuit:')
        self.log.info(circuit.qasm())
        self.log.info('test_teleport: data %s', data)
        self.log.info('test_teleport: alice %s', alice)
        self.log.info('test_teleport: bob %s', bob)
        alice_ratio = 1/np.tan(pi/8)**2
        bob_ratio = bob['0']/float(bob['1'])
        error = abs(alice_ratio - bob_ratio) / alice_ratio
        self.log.info('test_teleport: relative error = %s', error)
        self.assertLess(error, 0.05)


if __name__ == '__main__':
    unittest.main()
