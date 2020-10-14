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

""" Tests for circuit MPL drawer"""

import unittest

import json
import os
from contextlib import contextmanager
import math
import numpy as np
from numpy import pi

from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.test.mock import FakeTenerife
from qiskit.visualization.circuit_visualization import _matplotlib_circuit_drawer
from qiskit.circuit.library import XGate, MCXGate, HGate, RZZGate, SwapGate, DCXGate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit import Parameter
from qiskit.circuit.library import IQP
from qiskit.quantum_info.random import random_unitary

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


class TestMatplotlibDrawer(QiskitTestCase):
    """Circuit MPL visualization"""

    def setUp(self):
        super().setUp()
        self.circuit_drawer = TestMatplotlibDrawer.save_data_wrap(_matplotlib_circuit_drawer,
                                                                  str(self))

    @staticmethod
    def save_data_wrap(func, testname):
        """A wrapper to save the data from a test"""
        def wrapper(*args, **kwargs):
            image_filename = kwargs['filename']
            with cwd(RESULTDIR):
                results = func(*args, **kwargs)
                TestMatplotlibDrawer.save_data(image_filename, testname)
            return results

        return wrapper

    @staticmethod
    def save_data(image_filename, testname):
        """Saves result data of a test"""
        datafilename = 'result_test.json'
        if os.path.exists(datafilename):
            with open(datafilename) as datafile:
                data = json.load(datafile)
        else:
            data = {}
        data[image_filename] = {'testname': testname}
        with open(datafilename, 'w') as datafile:
            json.dump(data, datafile)

    def test_empty_circuit(self):
        """Test empty circuit"""
        circuit = QuantumCircuit()

        self.circuit_drawer(circuit, filename='empty_circut.png')

    def test_long_name(self):
        """Test to see that long register names can be seen completely
        As reported in #2605
        """

        # add a register with a very long name
        qr = QuantumRegister(4, 'veryLongQuantumRegisterName')
        # add another to make sure adjustments are made based on longest
        qrr = QuantumRegister(1, 'q0')
        circuit = QuantumCircuit(qr, qrr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.h(qr)
        circuit.h(qr)

        self.circuit_drawer(circuit, filename='long_name.png')

    def test_conditional(self):
        """Test that circuits with conditionals draw correctly
        """
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.measure(qr, cr)
        circuit.h(qr[0]).c_if(cr, 2)

        self.circuit_drawer(circuit, filename='conditional.png')

    def test_plot_partial_barrier(self):
        """Test plotting of partial barriers."""

        # generate a circuit with barrier and other barrier like instructions in
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier(0)
        circuit.h(q[0])

        self.circuit_drawer(circuit, filename='plot_partial_barrier.png', plot_barriers=True)

    def test_plot_barriers(self):
        """Test to see that plotting barriers works.
        If it is set to False, no blank columns are introduced"""

        # generate a circuit with barriers and other barrier like instructions in
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier()

        # check for other barrier like commands
        circuit.h(q[1])

        # this import appears to be unused, but is actually needed to get snapshot instruction
        import qiskit.extensions.simulator  # pylint: disable=unused-import
        circuit.snapshot('1')

        # check the barriers plot properly when plot_barriers= True
        self.circuit_drawer(circuit, filename='plot_barriers_true.png', plot_barriers=True)
        self.circuit_drawer(circuit, filename='plot_barriers_false.png', plot_barriers=False)

    def test_no_barriers_false(self):
        """Generate the same circuit as test_plot_barriers but without the barrier commands
         as this is what the circuit should look like when displayed with plot barriers false"""
        q1 = QuantumRegister(2, 'q')
        c1 = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(q1, c1)
        circuit.h(q1[0])
        circuit.h(q1[1])

        self.circuit_drawer(circuit, filename='no_barriers.png', plot_barriers=False)

    def test_fold_minus1(self):
        """Test to see that fold=-1 is no folding"""
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        for _ in range(3):
            circuit.h(0)
            circuit.x(0)

        self.circuit_drawer(circuit, fold=-1, filename='fold_minus1.png')

    def test_fold_4(self):
        """Test to see that fold=4 is folding"""
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)
        for _ in range(3):
            circuit.h(0)
            circuit.x(0)

        self.circuit_drawer(circuit, fold=4, filename='fold_4.png')

    def test_big_gates(self):
        """Test large gates with params"""
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.append(IQP([[6, 5, 3], [5, 4, 5], [3, 5, 1]]), [0, 1, 2])

        desired_vector = [
            1 / math.sqrt(16) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(16) * complex(1, 1),
            0,
            0,
            1 / math.sqrt(8) * complex(1, 2),
            1 / math.sqrt(16) * complex(1, 0),
            0]

        circuit.initialize(desired_vector, [qr[3], qr[4], qr[5]])
        circuit.unitary([[1, 0], [0, 1]], [qr[0]])
        matrix = np.zeros((4, 4))
        theta = Parameter('theta')
        circuit.append(HamiltonianGate(matrix, theta), [qr[1], qr[2]])
        circuit = circuit.bind_parameters({theta: 1})
        circuit.isometry(np.eye(4, 4), list(range(3, 5)), [])

        self.circuit_drawer(circuit, filename='big_gates.png')

    def test_cnot(self):
        """Test different cnot gates (ccnot, mcx, etc)"""
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.append(XGate().control(3, ctrl_state='010'), [qr[2], qr[3], qr[0], qr[1]])
        circuit.append(MCXGate(num_ctrl_qubits=3, ctrl_state='101'), [qr[0], qr[1], qr[2], qr[4]])

        self.circuit_drawer(circuit, filename='cnot.png')

    def test_pauli_clifford(self):
        """Test Pauli(green) and Clifford(blue) gates"""
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.y(0)
        circuit.z(0)
        circuit.id(0)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.cy(1, 2)
        circuit.cz(1, 2)
        circuit.swap(3, 4)
        circuit.s(3)
        circuit.sdg(3)
        circuit.iswap(3, 4)
        circuit.dcx(3, 4)

        self.circuit_drawer(circuit, filename='pauli_clifford.png')

    def test_u_gates(self):
        """Test U 1, 2, & 3 gates"""
        from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, CU1Gate, CU3Gate
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.append(U1Gate(3 * pi / 2), [0])
        circuit.append(U2Gate(3 * pi / 2, 2 * pi / 3), [1])
        circuit.append(U3Gate(3 * pi / 2, 4.5, pi / 4), [2])
        circuit.append(CU1Gate(pi / 4), [0, 1])
        circuit.append(U2Gate(pi / 2, 3 * pi / 2).control(1), [2, 3])
        circuit.append(CU3Gate(3 * pi / 2, -3 * pi / 4, -pi / 2), [0, 1])

        self.circuit_drawer(circuit, filename='u_gates.png')

    def test_creg_initial(self):
        """Test cregbundle and initial state options"""
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(0)
        circuit.h(0)
        circuit.x(1)

        self.circuit_drawer(circuit, filename='creg_initial_true.png',
                            cregbundle=True, initial_state=True)
        self.circuit_drawer(circuit, filename='creg_initial_false.png',
                            cregbundle=False, initial_state=False)

    def test_r_gates(self):
        """Test all R gates"""
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.r(3 * pi / 4, 3 * pi / 8, 0)
        circuit.rx(pi / 2, 1)
        circuit.ry(-pi / 2, 2)
        circuit.rz(3 * pi / 4, 3)
        circuit.rxx(pi / 2, 0, 1)
        circuit.ryy(3 * pi / 4, 2, 3)
        circuit.rzx(-pi / 2, 0, 1)
        circuit.rzz(pi / 2, 2, 3)

        self.circuit_drawer(circuit, filename='r_gates.png')

    def test_ctrl_labels(self):
        """Test control labels"""
        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cy(1, 0, label='Bottom Y Label')
        circuit.cy(2, 3, label='Top Y Label')
        circuit.ch(0, 1, label='Top H Label')
        circuit.append(HGate(label='H Gate Label').control(3, label='H Control Label',
                                                           ctrl_state='010'),
                       [qr[1], qr[2], qr[3], qr[0]])

        self.circuit_drawer(circuit, filename='ctrl_labels.png')

    def test_cswap_rzz(self):
        """Test controlled swap and rzz gates"""
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cswap(0, 1, 2)
        circuit.append(RZZGate(3 * pi / 4).control(3, ctrl_state='010'), [2, 1, 4, 3, 0])

        self.circuit_drawer(circuit, filename='cswap_rzz.png')

    def test_ghz_to_gate(self):
        """Test controlled GHZ to_gate circuit"""
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        ghz_circuit = QuantumCircuit(3, name='This is a WWWWWWWWWWWide name Ctrl-GHZ Circuit')
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2, ctrl_state='10')
        circuit.append(ccghz, [4, 0, 1, 3, 2])

        self.circuit_drawer(circuit, filename='ghz_to_gate.png')

    def test_scale(self):
        """Tests scale
        See: https://github.com/Qiskit/qiskit-terra/issues/4179"""
        circuit = QuantumCircuit(5)
        circuit.unitary(random_unitary(2 ** 5), circuit.qubits)

        self.circuit_drawer(circuit, filename='scale_default.png')
        self.circuit_drawer(circuit, filename='scale_half.png', scale=0.5)
        self.circuit_drawer(circuit, filename='scale_double.png', scale=2)

    def test_pi_param_expr(self):
        """Text pi in circuit with parameter expression."""
        x, y = Parameter('x'), Parameter('y')
        circuit = QuantumCircuit(1)
        circuit.rx((pi - x) * (pi - y), 0)

        self.circuit_drawer(circuit, filename='pi_in_param_expr.png')

    def test_partial_layout(self):
        """Tests partial_layout
        See: https://github.com/Qiskit/qiskit-terra/issues/4757"""
        circuit = QuantumCircuit(3)
        circuit.h(1)
        transpiled = transpile(circuit, backend=FakeTenerife(),
                               optimization_level=0, initial_layout=list(range(3)),
                               seed_transpiler=0)

        self.circuit_drawer(transpiled, filename='partial_layout.png')

    def test_init_reset(self):
        """Test reset and initialize with 1 and 2 qubits"""
        circuit = QuantumCircuit(2)
        circuit.initialize([0, 1], 0)
        circuit.reset(1)
        circuit.initialize([0, 1, 0, 0], [0, 1])

        self.circuit_drawer(circuit, filename='init_reset.png')

    def test_with_global_phase(self):
        """Tests with global phase"""
        circuit = QuantumCircuit(3, global_phase=1.57079632679)
        circuit.h(range(3))

        self.circuit_drawer(circuit, filename='global_phase.png')

    def test_iqx_colors(self):
        """Tests with iqx color scheme"""
        circuit = QuantumCircuit(7)
        circuit.h(0)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.swap(0, 1)
        circuit.cswap(0, 1, 2)
        circuit.append(SwapGate().control(2), [0, 1, 2, 3])
        circuit.dcx(0, 1)
        circuit.append(DCXGate().control(1), [0, 1, 2])
        circuit.append(DCXGate().control(2), [0, 1, 2, 3])
        circuit.z(4)
        circuit.s(4)
        circuit.sdg(4)
        circuit.t(4)
        circuit.tdg(4)
        circuit.p(pi/2, 4)
        circuit.u1(pi/2, 4)
        circuit.cz(5, 6)
        circuit.cu1(pi/2, 5, 6)
        circuit.y(5)
        circuit.rx(pi/3, 5)
        circuit.rzx(pi/2, 5, 6)
        circuit.u2(pi/2, pi/2, 5)
        circuit.barrier(5, 6)
        circuit.reset(5)

        self.circuit_drawer(circuit, style={'name': 'iqx'}, filename='iqx_color.png')


if __name__ == '__main__':
    unittest.main(verbosity=1)
