# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A test for visualizing device coupling maps"""
import unittest

from io import BytesIO
from PIL import Image
from ddt import ddt, data
from qiskit.test.mock import FakeProvider
from qiskit.visualization.gate_map import plot_gate_map, plot_circuit_layout
from qiskit.tools.visualization import HAS_MATPLOTLIB
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import Layout
from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt


@ddt
class TestGateMap(QiskitVisualizationTestCase):
    """visual tests for plot_gate_map"""

    backends = list(
        filter(
            lambda x: not x.configuration().simulator
            and x.configuration().num_qubits in range(5, 21),
            FakeProvider().backends(),
        )
    )

    @data(*backends)
    @unittest.skipIf(not HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_gate_map(self, backend):
        """tests plotting of gate map of a device (20 qubit, 16 qubit, 14 qubit and 5 qubit)"""
        n = backend.configuration().n_qubits
        img_ref = path_to_diagram_reference(str(n) + "bit_quantum_computer.png")
        fig = plot_gate_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)

    @data(*backends)
    @unittest.skipIf(not HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_circuit_layout(self, backend):
        """tests plot_circuit_layout for each device"""
        layout_length = int(backend._configuration.n_qubits / 2)
        qr = QuantumRegister(layout_length, "qr")
        circuit = QuantumCircuit(qr)
        circuit._layout = Layout({qr[i]: i * 2 for i in range(layout_length)})
        circuit._layout.add_register(qr)
        n = backend.configuration().n_qubits
        img_ref = path_to_diagram_reference(str(n) + "_plot_circuit_layout.png")
        fig = plot_circuit_layout(circuit, backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
