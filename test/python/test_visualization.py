import unittest
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

    # def test_get_image_depth(self):
    #     compiled_circuit = self.qobj['circuits'][0]['compiled_circuit']
    #     qcimg = QCircuitImage(compiled_circuit)
    #     self.assertEqual(qcimg._get_image_depth(), 6)

    def test_latex_drawer(self):
        filename = self._get_resource_path('test_latex_drawer.tex')
        try:
            latex_drawer(self.qc, filename)
        except:
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
        except:
            if os.path.exists(filename):
                os.remove(filename)
            raise

    # def test_latex_drawer_random(self):
    #     nCircuits = 20
    #     minDepth = 1
    #     maxDepth = 20
    #     minQubits = 1
    #     maxQubits = 5
    #     randomCircuits = RandomCircuitGenerator(seed=None,
    #                                             minQubits=minQubits,
    #                                             maxQubits=maxQubits,
    #                                             minDepth=minDepth,
    #                                             maxDepth=maxDepth)
    #     randomCircuits.add_circuits(nCircuits)
    #     qc_circuits = randomCircuits.get_circuits(format='QuantumCircuit')
    #     for i, circuit in enumerate(qc_circuits):
    #         filename = self._get_resource_path('latex_test' + str(i).zfill(3) +
    #                                            '.tex')
    #         print(circuit.qasm())
    #         latex_drawer(circuit, filename)

if __name__ == '__main__':
    unittest.main(verbosity=2)
