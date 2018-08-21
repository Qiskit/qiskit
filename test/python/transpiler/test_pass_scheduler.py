"""Tranpiler testing"""

from ..common import QiskitTestCase
import unittest.mock
import logging

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager, transpile, TransformationPass

logger = "LocalLogger"


class DummyTP(TransformationPass):
    def run(self, dag, property_set):
        logging.getLogger(logger).info('run pass %s', self.name)


class PassA(DummyTP):
    pass


class PassB(DummyTP):
    requires = [PassA()]
    preserves = [PassA()]


class TestUseCases(QiskitTestCase):
    def setUp(self):
        self.dag = DAGCircuit.fromQuantumCircuit(QuantumCircuit(QuantumRegister(1)))
        self.passmanager = PassManager()

    def assertScheduler(self, dag, passmanager, expected):
        """
        Runs transpiler(dag, passmanager) and checks if the passes run as expected.
        Args:
            dag (DAGCircuit): DAG circuit to transform via transpilation
            passmanager (PassManager): pass manager instance for the tranpilation process
            expected (list):
        """
        with self.assertLogs(logger, level='INFO') as cm:
            transpile(dag, pass_manager=passmanager)
        self.assertEqual([record.message for record in cm.records], expected)

    def test_do_not_repeat(self):
        self.passmanager.add_pass([PassB(), PassA(), PassB()])
        self.assertScheduler(self.dag, self.passmanager, ['run pass PassA',
                                                          'run pass PassB'])


if __name__ == '__main__':
    unittest.main()
