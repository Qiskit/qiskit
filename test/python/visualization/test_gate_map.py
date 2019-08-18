# -*- coding: utf-8 -*-

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
import os
from ddt import ddt, data
from qiskit.test.mock import FakeProvider
from qiskit.visualization.gate_map import plot_gate_map, plot_circuit_layout
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import Layout
from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase


@ddt
class TestGateMap(QiskitVisualizationTestCase):
    """ visual tests for plot_gate_map """
    backends = list(filter(lambda x:
                           not (x.configuration().simulator or x.configuration().n_qubits == 2),
                           FakeProvider().backends()))

    @data(*backends)
    def test_plot_gate_map(self, backend):
        """ tests plotting of gate map of a device (20 qubit, 16 qubit, 14 qubit and 5 qubit)"""
        n = backend.configuration().n_qubits
        img_ref = path_to_diagram_reference(str(n) + "bit_quantum_computer.png")
        filename = "temp.png"
        fig = plot_gate_map(backend)
        fig.savefig(filename)
        self.assertImagesAreEqual(filename, img_ref, 0.2)
        os.remove(filename)

    @data(*backends)
    def test_plot_circuit_layout(self, backend):
        """ tests plot_circuit_layout for each device"""
        layout_length = int(backend._configuration.n_qubits / 2)
        qr = QuantumRegister(layout_length, 'qr')
        circuit = QuantumCircuit(qr)
        circuit._layout = Layout({qr[i]: i * 2 for i in range(layout_length)})
        n = backend.configuration().n_qubits
        img_ref = path_to_diagram_reference(str(n) + "_plot_circuit_layout.png")
        filename = str(n) + "_plot_circuit_layout_result.png"
        fig = plot_circuit_layout(circuit, backend)
        fig.savefig(filename)
        self.assertImagesAreEqual(filename, img_ref, 0.1)
        os.remove(filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
