# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Tests for graph MPL drawer"""

import unittest

import json
import os
from contextlib import contextmanager

from qiskit.visualization.state_visualization import state_drawer
from qiskit import BasicAer, execute
from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit
from qiskit.tools.visualization import HAS_MATPLOTLIB
from qiskit.visualization.counts_visualization import plot_histogram
from qiskit.visualization.gate_map import plot_gate_map, plot_coupling_map
from qiskit.test.mock.fake_provider import (
    FakeArmonk,
    FakeBelem,
    FakeCasablanca,
    FakeRueschlikon,
    FakeMumbai,
    FakeManhattan,
)

if HAS_MATPLOTLIB:
    from matplotlib.pyplot import close as mpl_close
else:
    raise ImportError('Must have Matplotlib installed. To install, run "pip install matplotlib".')


RESULTDIR = os.path.dirname(os.path.abspath(__file__))


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
        self.graph_state_drawer = TestGraphMatplotlibDrawer.save_data_wrap(state_drawer, str(self))
        self.graph_count_drawer = TestGraphMatplotlibDrawer.save_data_wrap(
            plot_histogram, str(self)
        )
        self.graph_plot_gate_map = TestGraphMatplotlibDrawer.save_data_wrap(
            plot_gate_map, str(self)
        )
        self.graph_plot_coupling_map = TestGraphMatplotlibDrawer.save_data_wrap(
            plot_coupling_map, str(self)
        )

    def tearDown(self):
        super().tearDown()
        mpl_close("all")

    @staticmethod
    def save_data_wrap(func, testname):
        """A wrapper to save the data a test"""

        def wrapper(*args, **kwargs):
            image_filename = kwargs["filename"]
            with cwd(RESULTDIR):
                results = func(*args, **kwargs)
                TestGraphMatplotlibDrawer.save_data(image_filename, testname)
            return results

        return wrapper

    @staticmethod
    def save_data(image_filename, testname):
        """Saves result data of a test"""
        datafilename = "result_test.json"
        if os.path.exists(datafilename):
            with open(datafilename, encoding="UTF-8") as datafile:
                data = json.load(datafile)
        else:
            data = {}
        data[image_filename] = {"testname": testname}
        with open(datafilename, "w", encoding="UTF-8") as datafile:
            json.dump(data, datafile)

    def test_plot_bloch_multivector(self):
        """test bloch sphere
        See https://github.com/Qiskit/qiskit-terra/issues/6397.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)

        # getting the state using backend
        backend = BasicAer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        state = result.get_statevector(circuit)

        self.graph_state_drawer(state=state, output="bloch", filename="bloch_multivector.png")

    def test_plot_state_hinton(self):
        """test plot_state_hinton"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using backend
        backend = BasicAer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        state = result.get_statevector(circuit)

        self.graph_state_drawer(state=state, output="hinton", filename="hinton.png")

    def test_plot_state_qsphere(self):
        """test for plot_state_qsphere"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using backend
        backend = BasicAer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        state = result.get_statevector(circuit)

        self.graph_state_drawer(state=state, output="qsphere", filename="qsphere.png")

    def test_plot_state_city(self):
        """test for plot_state_city"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using backend
        backend = BasicAer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        state = result.get_statevector(circuit)

        self.graph_state_drawer(state=state, output="city", filename="state_city.png")

    def test_plot_state_paulivec(self):
        """test for plot_state_paulivec"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using backend
        backend = BasicAer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        state = result.get_statevector(circuit)

        self.graph_state_drawer(state=state, output="paulivec", filename="paulivec.png")

    def test_plot_histogram(self):
        """for testing the plot_histogram"""
        # specifing counts because we do not want oscillation of
        # result until a changes is made to plot_histogram

        counts = {"11": 500, "00": 500}

        self.graph_count_drawer(counts, filename="histogram.png")

    def test_plot_1_qubit_gate_map(self):
        """Test plot_gate_map using 1 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = FakeArmonk()

        self.graph_plot_gate_map(backend=backend, filename="1_qubit_gate_map.png")

    def test_plot_5_qubit_gate_map(self):
        """Test plot_gate_map using 5 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = FakeBelem()

        self.graph_plot_gate_map(backend=backend, filename="5_qubit_gate_map.png")

    def test_plot_7_qubit_gate_map(self):
        """Test plot_gate_map using 7 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = FakeCasablanca()

        self.graph_plot_gate_map(backend=backend, filename="7_qubit_gate_map.png")

    def test_plot_16_qubit_gate_map(self):
        """Test plot_gate_map using 16 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = FakeRueschlikon()

        self.graph_plot_gate_map(backend=backend, filename="16_qubit_gate_map.png")

    def test_plot_27_qubit_gate_map(self):
        """Test plot_gate_map using 27 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = FakeMumbai()

        self.graph_plot_gate_map(backend=backend, filename="27_qubit_gate_map.png")

    def test_plot_65_qubit_gate_map(self):
        """test for plot_gate_map using 65 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = FakeManhattan()

        self.graph_plot_gate_map(backend=backend, filename="65_qubit_gate_map.png")

    def test_figsize(self):
        """Test figsize parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = FakeBelem()

        self.graph_plot_gate_map(backend=backend, figsize=(10, 10), filename="figsize.png")

    def test_qubit_size(self):
        """Test qubit_size parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = FakeBelem()

        self.graph_plot_gate_map(backend=backend, qubit_size=38, filename="qubit_size.png")

    def test_qubit_color(self):
        """Test qubit_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = FakeCasablanca()

        self.graph_plot_gate_map(
            backend=backend, qubit_color=["#ff0000"] * 7, filename="qubit_color.png"
        )

    def test_qubit_labels(self):
        """Test qubit_labels parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = FakeCasablanca()

        self.graph_plot_gate_map(
            backend=backend, qubit_labels=list(range(10, 17, 1)), filename="qubit_labels.png"
        )

    def test_line_color(self):
        """Test line_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = FakeManhattan()

        self.graph_plot_gate_map(
            backend=backend, line_color=["#00ff00"] * 144, filename="line_color.png"
        )

    def test_font_color(self):
        """Test font_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = FakeManhattan()

        self.graph_plot_gate_map(backend=backend, font_color="#ff00ff", filename="font_color.png")

    def test_plot_coupling_map(self):
        """Test plot_coupling_map"""

        num_qubits = 5
        qubit_coordinates = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]
        coupling_map = [[1, 0], [1, 2], [1, 3], [3, 4]]

        self.graph_plot_coupling_map(
            num_qubits=num_qubits,
            qubit_coordinates=qubit_coordinates,
            coupling_map=coupling_map,
            filename="coupling_map.png",
        )


if __name__ == "__main__":
    unittest.main(verbosity=1)
