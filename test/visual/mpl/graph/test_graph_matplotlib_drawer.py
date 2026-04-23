# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for graph MPL drawer"""

import unittest
import os
from contextlib import contextmanager
from pathlib import Path

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit.utils import optionals
from qiskit.visualization.counts_visualization import plot_histogram
from qiskit.visualization.gate_map import (
    plot_circuit_layout,
    plot_coupling_map,
    plot_error_map,
    plot_gate_map,
)
from qiskit.visualization.state_visualization import state_drawer
from test import QiskitTestCase
from test.python.legacy_cmaps import (
    ALMADEN_CMAP,
    KYOTO_CMAP,
    YORKTOWN_CMAP,
    LAGOS_CMAP,
    RUESCHLIKON_CMAP,
    MUMBAI_CMAP,
    MANHATTAN_CMAP,
)
from test.visual import VisualTestUtilities

if optionals.HAS_MATPLOTLIB:
    from matplotlib.pyplot import close as mpl_close
else:
    raise ImportError('Must have Matplotlib installed. To install, run "pip install matplotlib".')

BASE_DIR = Path(__file__).parent
RESULT_DIR = Path(BASE_DIR) / "graph_results"
TEST_REFERENCE_DIR = Path(BASE_DIR) / "references"
FAILURE_DIFF_DIR = Path(BASE_DIR).parent / "visual_test_failures"
FAILURE_PREFIX = "graph_failure_"


@contextmanager
def cwd(path):
    """A context manager to run in a particular path"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class TestGraphMatplotlibDrawer(QiskitTestCase):
    """Graph MPL visualization"""

    def setUp(self):
        super().setUp()
        self.graph_state_drawer = VisualTestUtilities.save_data_wrap(
            state_drawer, str(self), RESULT_DIR
        )
        self.graph_count_drawer = VisualTestUtilities.save_data_wrap(
            plot_histogram, str(self), RESULT_DIR
        )
        self.graph_plot_gate_map = VisualTestUtilities.save_data_wrap(
            plot_gate_map, str(self), RESULT_DIR
        )
        self.graph_plot_coupling_map = VisualTestUtilities.save_data_wrap(
            plot_coupling_map, str(self), RESULT_DIR
        )

        if not os.path.exists(FAILURE_DIFF_DIR):
            os.makedirs(FAILURE_DIFF_DIR)

        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

    def tearDown(self):
        super().tearDown()
        mpl_close("all")

    @staticmethod
    def _image_path(image_name):
        return os.path.join(RESULT_DIR, image_name)

    @staticmethod
    def _reference_path(image_name):
        return os.path.join(TEST_REFERENCE_DIR, image_name)

    def _assert_figure_matches_reference(self, figure, image_name, threshold=0.99):
        with cwd(RESULT_DIR):
            figure.savefig(image_name)
            VisualTestUtilities.save_data(image_name, str(self))

        ratio = VisualTestUtilities._save_diff(
            self._image_path(image_name),
            self._reference_path(image_name),
            image_name,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, threshold, msg=image_name)

    def test_plot_bloch_multivector(self):
        """test bloch sphere
        See https://github.com/Qiskit/qiskit-terra/issues/6397.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "bloch_multivector.png"
        self.graph_state_drawer(state=state, output="bloch", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.98, msg=fname)

    def test_plot_state_hinton(self):
        """test plot_state_hinton"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "hinton.png"
        self.graph_state_drawer(state=state, output="hinton", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_state_qsphere(self):
        """test for plot_state_qsphere"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "qsphere.png"
        self.graph_state_drawer(state=state, output="qsphere", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_entangled_state_qsphere(self):
        """test for plot_state_qsphere"""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.z(0)
        circuit.h(0)
        circuit.cx(0, 1)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "entangled_qsphere.png"
        self.graph_state_drawer(state=state, output="qsphere", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_state_city(self):
        """test for plot_state_city"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "state_city.png"
        self.graph_state_drawer(state=state, output="city", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_state_paulivec(self):
        """test for plot_state_paulivec"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "paulivec.png"
        self.graph_state_drawer(state=state, output="paulivec", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram(self):
        """for testing the plot_histogram"""
        # specifying counts because we do not want oscillation of
        # result until a changes is made to plot_histogram

        counts = {"11": 500, "00": 500}

        fname = "histogram.png"
        self.graph_count_drawer(counts, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_with_rest(self):
        """test plot_histogram with 2 datasets and number_to_keep"""
        data = [{"00": 3, "01": 5, "10": 6, "11": 12}]

        fname = "histogram_with_rest.png"
        self.graph_count_drawer(data, number_to_keep=2, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_2_sets_with_rest(self):
        """test plot_histogram with 2 datasets and number_to_keep"""
        data = [
            {"00": 3, "01": 5, "10": 6, "11": 12},
            {"00": 5, "01": 7, "10": 6, "11": 12},
        ]

        fname = "histogram_2_sets_with_rest.png"
        self.graph_count_drawer(data, number_to_keep=2, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_color(self):
        """Test histogram with single color"""

        counts = {"00": 500, "11": 500}

        fname = "histogram_color.png"
        self.graph_count_drawer(data=counts, color="#204940", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_multiple_colors(self):
        """Test histogram with multiple custom colors"""

        counts = [
            {"00": 10, "01": 15, "10": 20, "11": 25},
            {"00": 25, "01": 20, "10": 15, "11": 10},
        ]

        fname = "histogram_multiple_colors.png"
        self.graph_count_drawer(
            data=counts,
            color=["#204940", "#c26219"],
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_hamming(self):
        """Test histogram with hamming distance"""

        counts = {"101": 500, "010": 500, "001": 500, "100": 500}

        fname = "histogram_hamming.png"
        self.graph_count_drawer(data=counts, sort="hamming", target_string="101", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_value_sort(self):
        """Test histogram with sorting by value"""

        counts = {"101": 300, "010": 240, "001": 80, "100": 150, "110": 160, "000": 280, "111": 60}

        fname = "histogram_value_sort.png"
        self.graph_count_drawer(data=counts, sort="value", target_string="000", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_desc_value_sort(self):
        """Test histogram with sorting by descending value"""

        counts = {"101": 150, "010": 50, "001": 180, "100": 10, "110": 190, "000": 80, "111": 260}

        fname = "histogram_desc_value_sort.png"
        self.graph_count_drawer(
            data=counts,
            sort="value_desc",
            target_string="000",
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_legend(self):
        """Test histogram with legend"""

        counts = [{"0": 50, "1": 30}, {"0": 30, "1": 40}]

        fname = "histogram_legend.png"
        self.graph_count_drawer(
            data=counts,
            legend=["first", "second"],
            filename=fname,
            figsize=(15, 5),
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_title(self):
        """Test histogram with title"""

        counts = [{"0": 50, "1": 30}, {"0": 30, "1": 40}]

        fname = "histogram_title.png"
        self.graph_count_drawer(data=counts, title="My Histogram", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_1_qubit_gate_map(self):
        """Test plot_gate_map using 1 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=1, basis_gates=["id", "rz", "sx", "x"])

        fname = "1_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_5_qubit_gate_map(self):
        """Test plot_gate_map using 5 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP)

        fname = "5_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_7_qubit_gate_map(self):
        """Test plot_gate_map using 7 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP)

        fname = "7_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_plot_16_qubit_gate_map(self):
        """Test plot_gate_map using 16 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=16, coupling_map=RUESCHLIKON_CMAP)

        fname = "16_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_plot_20_qubit_gate_map(self):
        """Test plot_gate_map using 20 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=20, coupling_map=ALMADEN_CMAP, seed=0)

        fname = "20_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.9, msg=fname)

    def test_plot_27_qubit_gate_map(self):
        """Test plot_gate_map using 27 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=27, coupling_map=MUMBAI_CMAP)

        fname = "27_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_65_qubit_gate_map(self):
        """test for plot_gate_map using 65 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=65, coupling_map=MANHATTAN_CMAP)

        fname = "65_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_figsize(self):
        """Test figsize parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP)

        fname = "figsize.png"
        self.graph_plot_gate_map(backend=backend, figsize=(10, 10), filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_qubit_size(self):
        """Test qubit_size parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP)

        fname = "qubit_size.png"
        self.graph_plot_gate_map(backend=backend, qubit_size=38, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_qubit_color(self):
        """Test qubit_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP)

        fname = "qubit_color.png"
        self.graph_plot_gate_map(backend=backend, qubit_color=["#ff0000"] * 7, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_qubit_labels(self):
        """Test qubit_labels parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP)

        fname = "qubit_labels.png"
        self.graph_plot_gate_map(
            backend=backend, qubit_labels=list(range(10, 17, 1)), filename=fname
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_line_color(self):
        """Test line_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=65, coupling_map=MANHATTAN_CMAP)

        fname = "line_color.png"
        self.graph_plot_gate_map(backend=backend, line_color=["#00ff00"] * 144, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_font_color(self):
        """Test font_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=65, coupling_map=MANHATTAN_CMAP)

        fname = "font_color.png"
        self.graph_plot_gate_map(backend=backend, font_color="#ff00ff", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_coupling_map(self):
        """Test plot_coupling_map"""

        num_qubits = 5
        qubit_coordinates = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]
        coupling_map = [[1, 0], [1, 2], [1, 3], [3, 4]]

        fname = "coupling_map.png"
        self.graph_plot_coupling_map(
            num_qubits=num_qubits,
            qubit_coordinates=qubit_coordinates,
            coupling_map=coupling_map,
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_coupling_map_no_backend(self):
        """Test plot_coupling_map without a backend"""

        num_qubits = 8
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
        qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]

        fname = "8_qubit_coupling_map.png"
        self.graph_plot_coupling_map(
            num_qubits=num_qubits,
            qubit_coordinates=qubit_coordinates,
            coupling_map=coupling_map,
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.8, msg=fname)

    def test_plot_5_qubit_circuit_layout(self):
        """Test plot_circuit_layout using a 5 qubit backend"""

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP, seed=0)
        layout_length = int(backend.num_qubits / 2)
        qr = QuantumRegister(layout_length, "qr")
        circuit = QuantumCircuit(qr)
        circuit._layout = TranspileLayout(
            Layout({qr[i]: i * 2 for i in range(layout_length)}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        circuit._layout.initial_layout.add_register(qr)

        fname = "5_plot_circuit_layout.png"
        figure = plot_circuit_layout(circuit, backend)
        self._assert_figure_matches_reference(figure, fname, threshold=0.9)

    def test_plot_7_qubit_circuit_layout(self):
        """Test plot_circuit_layout using a 7 qubit backend"""

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP, seed=0)
        layout_length = int(backend.num_qubits / 2)
        qr = QuantumRegister(layout_length, "qr")
        circuit = QuantumCircuit(qr)
        circuit._layout = TranspileLayout(
            Layout({qr[i]: i * 2 for i in range(layout_length)}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        circuit._layout.initial_layout.add_register(qr)

        fname = "7_plot_circuit_layout.png"
        figure = plot_circuit_layout(circuit, backend)
        self._assert_figure_matches_reference(figure, fname, threshold=0.9)

    def test_plot_20_qubit_circuit_layout(self):
        """Test plot_circuit_layout using a 20 qubit backend"""

        backend = GenericBackendV2(num_qubits=20, coupling_map=ALMADEN_CMAP, seed=0)
        layout_length = int(backend.num_qubits / 2)
        qr = QuantumRegister(layout_length, "qr")
        circuit = QuantumCircuit(qr)
        circuit._layout = TranspileLayout(
            Layout({qr[i]: i * 2 for i in range(layout_length)}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        circuit._layout.initial_layout.add_register(qr)

        fname = "20_plot_circuit_layout.png"
        figure = plot_circuit_layout(circuit, backend)
        self._assert_figure_matches_reference(figure, fname, threshold=0.9)

    def test_plot_error_map_backend_v2(self):
        """Test plotting error map with fake backend v2."""

        backend = GenericBackendV2(num_qubits=27, coupling_map=MUMBAI_CMAP)

        fname = "fake_27_q_v2_error.png"
        figure = plot_error_map(backend)
        self._assert_figure_matches_reference(figure, fname, threshold=0.95)

    def test_plot_error_map_over_100_qubit(self):
        """Test plotting error map with large fake backend."""

        backend = GenericBackendV2(num_qubits=127, coupling_map=KYOTO_CMAP, seed=42)

        fname = "fake_127_q_error.png"
        figure = plot_error_map(backend)
        self._assert_figure_matches_reference(figure, fname, threshold=0.95)

    def test_plot_error_map_over_100_qubit_backend_v2(self):
        """Test plotting error map with large fake backend v2."""

        coupling_map = [
            [0, 1],
            [0, 14],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 5],
            [4, 15],
            [5, 4],
            [5, 6],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [8, 7],
            [8, 16],
            [9, 10],
            [10, 9],
            [10, 11],
            [11, 10],
            [11, 12],
            [12, 11],
            [12, 13],
            [12, 17],
            [13, 12],
            [14, 0],
            [14, 18],
            [15, 4],
            [15, 22],
            [16, 8],
            [16, 26],
            [17, 12],
            [17, 30],
            [18, 14],
            [18, 19],
            [19, 18],
            [19, 20],
            [20, 19],
            [20, 21],
            [20, 33],
            [21, 20],
            [21, 22],
            [22, 15],
            [22, 21],
            [22, 23],
            [23, 22],
            [23, 24],
            [24, 23],
            [24, 25],
            [24, 34],
            [25, 24],
            [25, 26],
            [26, 16],
            [26, 25],
            [26, 27],
            [27, 26],
            [27, 28],
            [28, 27],
            [28, 29],
            [28, 35],
            [29, 28],
            [29, 30],
            [30, 17],
            [30, 29],
            [30, 31],
            [31, 30],
            [31, 32],
            [32, 31],
            [32, 36],
            [33, 20],
            [33, 39],
            [34, 24],
            [34, 43],
            [35, 28],
            [35, 47],
            [36, 32],
            [36, 51],
            [37, 38],
            [37, 52],
            [38, 37],
            [38, 39],
            [39, 33],
            [39, 38],
            [39, 40],
            [40, 39],
            [40, 41],
            [41, 40],
            [41, 42],
            [41, 53],
            [42, 41],
            [42, 43],
            [43, 34],
            [43, 42],
            [43, 44],
            [44, 43],
            [44, 45],
            [45, 44],
            [45, 46],
            [45, 54],
            [46, 45],
            [46, 47],
            [47, 35],
            [47, 46],
            [47, 48],
            [48, 47],
            [48, 49],
            [49, 48],
            [49, 50],
            [49, 55],
            [50, 49],
            [50, 51],
            [51, 36],
            [51, 50],
            [52, 37],
            [52, 56],
            [53, 41],
            [53, 60],
            [54, 45],
            [54, 64],
            [55, 49],
            [55, 68],
            [56, 52],
            [56, 57],
            [57, 56],
            [57, 58],
            [58, 57],
            [58, 59],
            [58, 71],
            [59, 58],
            [59, 60],
            [60, 53],
            [60, 59],
            [60, 61],
            [61, 60],
            [61, 62],
            [62, 61],
            [62, 63],
            [62, 72],
            [63, 62],
            [63, 64],
            [64, 54],
            [64, 63],
            [64, 65],
            [65, 64],
            [65, 66],
            [66, 65],
            [66, 67],
            [66, 73],
            [67, 66],
            [67, 68],
            [68, 55],
            [68, 67],
            [68, 69],
            [69, 68],
            [69, 70],
            [70, 69],
            [70, 74],
            [71, 58],
            [71, 77],
            [72, 62],
            [72, 81],
            [73, 66],
            [73, 85],
            [74, 70],
            [74, 89],
            [75, 76],
            [75, 90],
            [76, 75],
            [76, 77],
            [77, 71],
            [77, 76],
            [77, 78],
            [78, 77],
            [78, 79],
            [79, 78],
            [79, 80],
            [79, 91],
            [80, 79],
            [80, 81],
            [81, 72],
            [81, 80],
            [81, 82],
            [82, 81],
            [82, 83],
            [83, 82],
            [83, 84],
            [83, 92],
            [84, 83],
            [84, 85],
            [85, 73],
            [85, 84],
            [85, 86],
            [86, 85],
            [86, 87],
            [87, 86],
            [87, 88],
            [87, 93],
            [88, 87],
            [88, 89],
            [89, 74],
            [89, 88],
            [90, 75],
            [90, 94],
            [91, 79],
            [91, 98],
            [92, 83],
            [92, 102],
            [93, 87],
            [93, 106],
            [94, 90],
            [94, 95],
            [95, 94],
            [95, 96],
            [96, 95],
            [96, 97],
            [96, 109],
            [97, 96],
            [97, 98],
            [98, 91],
            [98, 97],
            [98, 99],
            [99, 98],
            [99, 100],
            [100, 99],
            [100, 101],
            [100, 110],
            [101, 100],
            [101, 102],
            [102, 92],
            [102, 101],
            [102, 103],
            [103, 102],
            [103, 104],
            [104, 103],
            [104, 105],
            [104, 111],
            [105, 104],
            [105, 106],
            [106, 93],
            [106, 105],
            [106, 107],
            [107, 106],
            [107, 108],
            [108, 107],
            [108, 112],
            [109, 96],
            [110, 100],
            [110, 118],
            [111, 104],
            [111, 122],
            [112, 108],
            [112, 126],
            [113, 114],
            [114, 113],
            [114, 115],
            [115, 114],
            [115, 116],
            [116, 115],
            [116, 117],
            [117, 116],
            [117, 118],
            [118, 110],
            [118, 117],
            [118, 119],
            [119, 118],
            [119, 120],
            [120, 119],
            [120, 121],
            [121, 120],
            [121, 122],
            [122, 111],
            [122, 121],
            [122, 123],
            [123, 122],
            [123, 124],
            [124, 123],
            [124, 125],
            [125, 124],
            [125, 126],
            [126, 112],
            [126, 125],
        ]

        backend = GenericBackendV2(num_qubits=127, coupling_map=coupling_map, seed=42)

        fname = "fake_127_q_v2_error.png"
        figure = plot_error_map(backend)
        self._assert_figure_matches_reference(figure, fname, threshold=0.8)

    def test_plot_bloch_multivector_figsize_improvements(self):
        """test bloch sphere figsize, font_size, title_font_size and title_pad
        See https://github.com/Qiskit/qiskit-terra/issues/7263
        and https://github.com/Qiskit/qiskit-terra/pull/7264.
        """
        circuit = QuantumCircuit(3)
        circuit.h(1)
        circuit.sxdg(2)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "bloch_multivector_figsize_improvements.png"
        self.graph_state_drawer(
            state=state,
            output="bloch",
            figsize=(3, 2),
            font_size=10,
            title="|0+R> state",
            title_font_size=14,
            title_pad=8,
            filename=fname,
        )
        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.98, msg=fname)


if __name__ == "__main__":
    unittest.main(verbosity=1)
