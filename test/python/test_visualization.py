import os
from qiskit import QuantumProgram
from .common import QiskitTestCase, TRAVIS_FORK_PULL_REQUEST, Path
from qiskit.tools.visualization import QCircuitImage, latex_drawer
from ._random_circuit_generator import RandomCircuitGenerator


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

    def test_get_image_depth(self):
        qcimg = QCircuitImage(self.qc)
        self.assertEqual(qcimg._get_image_depth(), 7)

    def test_latex_drawer(self):
        filename = self._get_resource_path('test_latex_drawer.tex')
        try:
            latex_drawer(self.qc, filename)
        except:
            if os.path.exists(filename):
                os.remove(filename)
            raise

    def test_latex_drawer_random(self):
        nCircuits = 20
        minDepth = 1
        maxDepth = 20
        minQubits = 1
        maxQubits = 5
        randomCircuits = RandomCircuitGenerator(seed=None,
                                                minQubits=minQubits,
                                                maxQubits=maxQubits,
                                                minDepth=minDepth,
                                                maxDepth=maxDepth)
        randomCircuits.add_circuits(nCircuits)
        qc_circuits = randomCircuits.get_circuits(format='QuantumCircuit')
        for i, circuit in enumerate(qc_circuits):
            filename = self._get_resource_path('latex_test' + str(i).zfill(3) +
                                               '.tex')
            latex_drawer(circuit, filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
