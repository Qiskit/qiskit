import unittest
import numpy
from scipy.stats import chi2_contingency
import qiskit
from qiskit import (ClassicalRegister, QuantumCircuit, QuantumProgram,
                    QuantumRegister)
from qiskit import QuantumJob
from qiskit._qiskiterror import QISKitError
from qiskit._compiler import compile, compile_circuit
from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.ibmq import IBMQJob
from .common import requires_qe_access, QiskitTestCase

class TestIBMQJob(QiskitTestCase):
    """
    Test ibmqjob module.
    """

    @classmethod
    @requires_qe_access
    def setUpClass(cls, QE_TOKEN, QE_URL):
        super().setUpClass()
        # create QuantumCircuit
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)
        cls._qc = qc
        cls._provider = IBMQProvider(QE_TOKEN, QE_URL)

    def test_run(self):
        backend = self._provider.get_backend('ibmqx_qasm_simulator')
        qobj = compile(self._qc, backend)
        quantum_job = QuantumJob(qobj, backend, shots=1024, preformatted=True)
        job = backend.run(quantum_job)
        result = job.result()
        counts_qx = result.get_counts(result.get_names()[0])
        counts_ex = {'00': 512, '11':512}
        states = counts_qx.keys() | counts_ex.keys()
        # contingency table
        ctable = numpy.array([[counts_qx.get(key, 0) for key in states],
                              [counts_ex.get(key, 0) for key in states]])
        contingency = chi2_contingency(ctable)
        self.log.info('chi2_contingency: %s', str(contingency))
        self.assertGreater(contingency[1], 0.01)

    @unittest.skip('cancel is not currently possible on IBM Q but should be soon')
    def test_cancel(self):
        backend = self._provider.get_backend('ibmqx4')
        qobj = compile(self._qc, backend)        
        quantum_job = QuantumJob(qobj, backend, shots=1024, preformatted=True)
        job = backend.run(quantum_job)
        job.cancel()
        self.assertTrue(job.cancelled)

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
