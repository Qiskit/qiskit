# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin


"""BasicAer provider integration tests (compile and run)."""

import unittest

from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile, execute
from qiskit.result import Result
from qiskit.test import QiskitTestCase


class TestBasicAerIntegration(QiskitTestCase):
    """Qiskit BasicAer simulator integration tests."""

    def setUp(self):
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self._qc1 = QuantumCircuit(qr, cr, name='qc1')
        self._qc2 = QuantumCircuit(qr, cr, name='qc2')
        self._qc1.measure(qr[0], cr[0])
        self.backend = BasicAer.get_backend('qasm_simulator')
        self._result1 = execute(self._qc1, self.backend).result()

    def test_builtin_simulator_result_fields(self):
        """Test components of a result from a local simulator."""

        self.assertEqual('qasm_simulator', self._result1.backend_name)
        self.assertIsInstance(self._result1.job_id, str)
        self.assertEqual(self._result1.status, 'COMPLETED')
        self.assertEqual(self._result1.results[0].status, 'DONE')

    def test_compile_run(self):
        """Test Compiler and run."""
        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = compile(qc, self.backend)
        result = self.backend.run(qobj).result()
        self.assertIsInstance(result, Result)

    def test_compile_two_run(self):
        """Test Compiler and run."""
        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = compile([qc, qc_extra], self.backend)
        result = self.backend.run(qobj).result()
        self.assertIsInstance(result, Result)

    def test_execute(self):
        """Test Execute."""
        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        job = execute(qc, self.backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    def test_execute_two(self):
        """Test execute two."""
        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], self.backend)
        results = job.result()
        self.assertIsInstance(results, Result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
