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
from qiskit.providers.fake_provider import (
    FakeProvider,
    FakeKolkata,
    FakeWashington,
    FakeKolkataV2,
    FakeWashingtonV2,
)
from qiskit.visualization import (
    plot_gate_map,
    plot_coupling_map,
    plot_circuit_layout,
    plot_error_map,
)
from qiskit.utils import optionals
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.layout import Layout, TranspileLayout
from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase

if optionals.HAS_MATPLOTLIB:
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
    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
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
    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_circuit_layout(self, backend):
        """tests plot_circuit_layout for each device"""
        layout_length = int(backend._configuration.n_qubits / 2)
        qr = QuantumRegister(layout_length, "qr")
        circuit = QuantumCircuit(qr)
        circuit._layout = TranspileLayout(
            Layout({qr[i]: i * 2 for i in range(layout_length)}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        circuit._layout.initial_layout.add_register(qr)
        n = backend.configuration().n_qubits
        img_ref = path_to_diagram_reference(str(n) + "_plot_circuit_layout.png")
        fig = plot_circuit_layout(circuit, backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_gate_map_no_backend(self):
        """tests plotting of gate map without a device"""
        n_qubits = 8
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
        qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]
        img_ref = path_to_diagram_reference(str(n_qubits) + "qubits.png")
        fig = plot_coupling_map(
            num_qubits=n_qubits, qubit_coordinates=qubit_coordinates, coupling_map=coupling_map
        )
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_error_map_backend_v1(self):
        """Test plotting error map with fake backend v1."""
        backend = FakeKolkata()
        img_ref = path_to_diagram_reference("kolkata_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_error_map_backend_v2(self):
        """Test plotting error map with fake backend v2."""
        backend = FakeKolkataV2()
        img_ref = path_to_diagram_reference("kolkata_v2_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_error_map_over_100_qubit(self):
        """Test plotting error map with large fake backend."""
        backend = FakeWashington()
        img_ref = path_to_diagram_reference("washington_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_plot_error_map_over_100_qubit_backend_v2(self):
        """Test plotting error map with large fake backendv2."""
        backend = FakeWashingtonV2()
        img_ref = path_to_diagram_reference("washington_v2_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
