# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Tests for comparing the outputs of circuit drawer with expected ones."""

import os
import unittest
from codecs import encode
from math import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, CU1Gate, CU3Gate

from qiskit.tools.visualization import HAS_MATPLOTLIB, circuit_drawer
from qiskit.visualization.circuit_visualization import _text_circuit_drawer

from .visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class TestCircuitVisualizationImplementation(QiskitVisualizationTestCase):
    """Visual accuracy of visualization tools outputs tests."""

    latex_reference = path_to_diagram_reference('circuit_latex_ref.png')
    matplotlib_reference = path_to_diagram_reference('circuit_matplotlib_ref.png')
    text_reference_utf8 = path_to_diagram_reference('circuit_text_ref_utf8.txt')
    text_reference_cp437 = path_to_diagram_reference('circuit_text_ref_cp437.txt')

    def sample_circuit(self):
        """Generate a sample circuit that includes the most common elements of
        quantum circuits.
        """
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.z(qr[0])
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])
        circuit.barrier(qr[2])
        circuit.h(qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[0])
        circuit.t(qr[0])
        circuit.tdg(qr[0])
        circuit.sx(qr[0])
        circuit.sxdg(qr[0])
        circuit.i(qr[0])
        circuit.reset(qr[0])
        circuit.rx(pi, qr[0])
        circuit.ry(pi, qr[0])
        circuit.rz(pi, qr[0])
        circuit.append(U1Gate(pi), [qr[0]])
        circuit.append(U2Gate(pi, pi), [qr[0]])
        circuit.append(U3Gate(pi, pi, pi), [qr[0]])
        circuit.swap(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.ch(qr[0], qr[1])
        circuit.append(CU1Gate(pi), [qr[0], qr[1]])
        circuit.append(CU3Gate(pi, pi, pi), [qr[0], qr[1]])
        circuit.crz(pi, qr[0], qr[1])
        circuit.cry(pi, qr[0], qr[1])
        circuit.crx(pi, qr[0], qr[1])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.measure(qr, cr)

        return circuit

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_latex_drawer(self):
        filename = 'current_latex.png'
        qc = self.sample_circuit()
        circuit_drawer(qc, filename=filename, output='latex')
        self.assertImagesAreEqual(filename, self.latex_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_matplotlib_drawer(self):
        filename = 'current_matplot.png'
        qc = self.sample_circuit()
        circuit_drawer(qc, filename=filename, output='mpl')
        self.assertImagesAreEqual(filename, self.matplotlib_reference)
        os.remove(filename)

    def test_text_drawer_utf8(self):
        filename = 'current_textplot_utf8.txt'
        qc = self.sample_circuit()
        output = _text_circuit_drawer(qc, filename=filename, fold=-1,
                                      initial_state=True, cregbundle=False, encoding='utf8')
        try:
            encode(str(output), encoding='utf8')
        except UnicodeEncodeError:
            self.fail("_text_circuit_drawer() should be utf8.")
        self.assertFilesAreEqual(filename, self.text_reference_utf8, 'utf8')
        os.remove(filename)

    def test_text_drawer_cp437(self):
        filename = 'current_textplot_cp437.txt'
        qc = self.sample_circuit()
        output = _text_circuit_drawer(qc, filename=filename, fold=-1,
                                      initial_state=True, cregbundle=False, encoding='cp437')
        try:
            encode(str(output), encoding='cp437')
        except UnicodeEncodeError:
            self.fail("_text_circuit_drawer() should be cp437.")
        self.assertFilesAreEqual(filename, self.text_reference_cp437, 'cp437')
        os.remove(filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
