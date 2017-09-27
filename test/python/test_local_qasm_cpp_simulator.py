import unittest
import os
import logging
import qiskit
import qiskit.backends._qasm_cpp_simulator as qasmcppsimulator
import qiskit._jobprocessor as jobprocessor
from qiskit import QuantumProgram
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import _openquantumcompiler as openquantumcompiler

class TestLocalQasmCppSimulator(unittest.TestCase):
    """
    Test job_pocessor module.
    """

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
        with open(self.qasmFileName, 'r') as qasm_file:
            self.qasm_text = qasm_file.read()
        qr = QuantumRegister('q', 2)
        cr = ClassicalRegister('c', 2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qc = qc
        # create qobj
        compiled_circuit1 = openquantumcompiler.compile(self.qc.qasm(),
                                                        format='json')
        compiled_circuit2 = openquantumcompiler.compile(self.qasm_text,
                                                        format='json')
        self.qobj = {'id': 'test_qobj',
                     'config': {
                         'max_credits': 3,
                         'shots': 100,
                         'backend': 'local_qasm_simulator',
                         'seed': 1111
                     },
                     'circuits': [
                         {
                             'name': 'test_circuit1',
                             'compiled_circuit': compiled_circuit1,
                             'basis_gates': 'u1,u2,u3,cx,id',
                             'layout': None,
                         },
                         {
                             'name': 'test_circuit2',
                             'compiled_circuit': compiled_circuit2,
                             'basis_gates': 'u1,u2,u3,cx,id',
                             'layout': None,
                         }
                     ]
                     }
        

    def test_run_qobj(self):
        try:
            simulator = qasmcppsimulator.QasmCppSimulator(self.qobj)
        except FileNotFoundError as fnferr:
            raise unittest.SkipTest(
                'cannot find {} in path'.format(fnferr))
        result = simulator.run()
        expected2 = {'000 000': 14,
                     '001 001': 12,
                     '010 010': 10,
                     '011 011': 12,
                     '100 100': 18,
                     '101 101': 9,
                     '110 110': 5,
                     '111 111': 20}
        self.assertEqual(result.get_counts('test_circuit2'), expected2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
