#!/usr/bin/env python
import unittest
import time
import numpy as np
import os
import sys
import cProfile
import pstats
import io
import logging
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.ticker import MaxNLocator
try:
    import qiskit
except ImportError as ierr:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    import qiskit
from qiskit import QuantumProgram
from qiskit.simulators._unitarysimulator import UnitarySimulator
import qiskit.qasm as qasm
import qiskit.unroll as unroll
if __name__ == '__main__':
    from _random_qasm_generator import RandomQasmGenerator
else:
    from test.python._random_qasm_generator import RandomQasmGenerator
import json

class LocalUnitarySimulatorTest(unittest.TestCase):
    """Test local unitary simulator."""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(__file__)[0]
        cls.log = logging.getLogger(__name__)
        cls.log.setLevel(logging.INFO)
        logFileName = cls.moduleName + '.log'
        handler = logging.FileHandler(logFileName)
        handler.setLevel(logging.INFO)
        log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                   ' %(message)s'.format(cls.__name__))
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        cls.log.addHandler(handler)

    def setUp(self):
        self.seed = 88
        self.qasmFileName = os.path.join(qiskit.__path__[0],
                                         '../test/python/qasm/example.qasm')
        self.modulePath = os.path.dirname(__file__)
        self.qp = QuantumProgram()

    def tearDown(self):
        pass

    def test_unitary_simulator(self):
        """test generation of circuit unitary"""
        shots = 1024
        self.qp.load_qasm_file(self.qasmFileName, name='example')
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(
            qasm.Qasm(data=self.qp.get_qasm('example')).parse(),
                      unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
	# if we want to manipulate the circuit, we have to convert it to a dict
        circuit = json.loads(circuit.decode())
        #strip measurements from circuit to avoid warnings
        circuit['operations'] = [op for op in circuit['operations']
                                 if op['name'] != 'measure']
	# the simulator is expecting a JSON format, so we need to convert it back to JSON
        qobj = {'id': 'unitary',
                'config': {
                    'max_credits': None,
                    'shots': 1,
                    'backend': 'local_unitary_simulator'
                    },
                'circuits': [
                    {
                        'name': 'test',
                        'compiled_circuit': json.dumps(circuit).encode(),
                        'compiled_circuit_qasm': self.qp.get_qasm('example'),
                        'config': {
                            'coupling_map': None,
                            'basis_gates': None,
                            'layout': None,
                            'seed': None
                            }
                    }
                ]
        }
        # numpy.savetxt currently prints complex numbers in a way
        # loadtxt can't read. To save file do,
        # fmtstr=['% .4g%+.4gj' for i in range(numCols)]
        # np.savetxt('example_unitary_matrix.dat', numpyMatrix, fmt=fmtstr, delimiter=',')
        expected = np.loadtxt(os.path.join(self.modulePath,
                                           'example_unitary_matrix.dat'),
                              dtype='complex', delimiter=',')
        result = UnitarySimulator(qobj).run()
        self.assertTrue(np.allclose(result.get_data('test')['unitary'],
                                    expected,
                                    rtol=1e-3))

    def profile_unitary_simulator(self):
        """Profile randomly generated circuits.

        Writes profile results to <this_module>.prof as well as recording
        to the log file.

        number of circuits = 100.
        number of operations/circuit in [1, 40]
        number of qubits in [1, 5]
        """
        nCircuits = 100
        maxDepth = 40
        maxQubits = 5
        pr = cProfile.Profile()
        randomCircuits = RandomQasmGenerator(seed=self.seed,
                                             maxDepth=maxDepth,
                                             maxQubits=maxQubits)
        randomCircuits.add_circuits(nCircuits, doMeasure=False)
        self.qp = randomCircuits.getProgram()
        pr.enable()
        self.qp.execute(self.qp.get_circuit_names(),
                        backend='local_unitary_simulator')
        pr.disable()
        sout = io.StringIO()
        ps = pstats.Stats(pr, stream=sout).sort_stats('cumulative')
        self.log.info('------- start profiling UnitarySimulator -----------')
        ps.print_stats()
        self.log.info(sout.getvalue())
        self.log.info('------- stop profiling UnitarySimulator -----------')
        sout.close()
        pr.dump_stats(self.moduleName + '.prof')

if __name__ == '__main__':
    unittest.main()
