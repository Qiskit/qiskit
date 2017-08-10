#!/usr/bin/env python
import unittest
import time
import os
import sys
import io
import logging
import random
import qiskit
from qiskit import QuantumProgram
import qiskit.qasm as qasm
import qiskit.unroll as unroll

from qiskit.simulators import _localsimulator

class LocalSimulatorTest(unittest.TestCase):
    """
    Test interface to local simulators.
    """

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.logFileName = cls.moduleName + '.log'
        log_fmt = 'LocalSimulatorTest:%(levelname)s:%(asctime)s: %(message)s'
        logging.basicConfig(filename=cls.logFileName, level=logging.INFO,
                            format=log_fmt)

    @classmethod
    def tearDownClass(cls):
        #cls.pdf.close()
        pass

    def setUp(self):
        self.seed = 88
        self.qasmFileName = os.path.join(qiskit.__path__[0],
                                         '../test/python/qasm/example.qasm')
        self.qp = QuantumProgram()
        shots = 1
        self.qp.load_qasm_file(self.qasmFileName, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm("example")).parse(),
                      unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        self.job = {'compiled_circuit': circuit,
                    'config': {'shots': shots, 'seed': random.randint(0, 10)}
                   }

    def tearDown(self):
        pass

    def test_local_configuration_present(self):
        self.assertTrue(_localsimulator.local_configuration)

    def test_local_configurations(self):
        required_keys = ['name',
                         'url',
                         'simulator',
                         'description',
                         'coupling_map',
                         'basis_gates']
        for conf in _localsimulator.local_configuration:
            for key in required_keys:
                self.assertIn(key, conf.keys())

    def test_simulator_classes(self):
        cdict = _localsimulator._simulator_classes
        cdict = getattr(_localsimulator, '_simulator_classes')
        logging.info('found local simulators: {0}'.format(repr(cdict)))
        self.assertTrue(cdict)

    def test_local_backends(self):
        backends = _localsimulator.local_backends()
        logging.info('found local backends: {0}'.format(repr(backends)))
        self.assertTrue(backends)

    def test_instantiation(self):
        """
        Test instantiation of LocalSimulator
        """
        backend_list = _localsimulator.local_backends()
        for backend_name in backend_list:
            backend = _localsimulator.LocalSimulator(backend_name, self.job)

if __name__ == '__main__':
    unittest.main()
