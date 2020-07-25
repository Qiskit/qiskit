# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests basic functionality of the transpile function"""

import io
import sys
import math

from logging import StreamHandler, getLogger
from unittest.mock import patch

from ddt import ddt, data, unpack

from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.circuit.library import CXGate
from qiskit.test import QiskitTestCase, Path
from qiskit.test.mock import FakeMelbourne, FakeRueschlikon
from qiskit.transpiler import Layout, CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements, CXDirection
from qiskit.quantum_info import Operator
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager


@ddt
class TestTranspile(QiskitTestCase):
    """Test transpile function."""

    def test_pass_manager_none(self):
        """Test passing the default (None) pass manager to the transpiler.

        It should perform the default qiskit flow:
        unroll, swap_mapper, cx_direction, cx_cancellation, optimize_1q_gates
        and should be equivalent to using tools.compile
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])

        coupling_map = [[1, 0]]
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']

        backend = BasicAer.get_backend('qasm_simulator')
        circuit2 = transpile(circuit, backend=backend, coupling_map=coupling_map,
                             basis_gates=basis_gates, pass_manager=None)

        circuit3 = transpile(circuit, backend=backend, coupling_map=coupling_map,
                             basis_gates=basis_gates)
        self.assertEqual(circuit2, circuit3)

    def test_transpile_basis_gates_no_backend_no_coupling_map(self):
        """Verify transpile() works with no coupling_map or backend."""
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']
        circuit2 = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
        resources_after = circuit2.count_ops()
        self.assertEqual({'u2': 2, 'cx': 4}, resources_after)

    def test_transpile_non_adjacent_layout(self):
        """Transpile pipeline can handle manual layout on non-adjacent qubits.

        circuit:
        qr0:-[H]--.------------  -> 1
                  |
        qr1:-----(+)--.--------  -> 2
                      |
        qr2:---------(+)--.----  -> 3
                          |
        qr3:-------------(+)---  -> 5

        device:
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])

        coupling_map = FakeMelbourne().configuration().coupling_map
        basis_gates = FakeMelbourne().configuration().basis_gates
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        new_circuit = transpile(circuit,
                                basis_gates=basis_gates,
                                coupling_map=coupling_map,
                                initial_layout=initial_layout)

        for gate, qargs, _ in new_circuit.data:
            if isinstance(gate, CXGate):
                self.assertIn([x.index for x in qargs], coupling_map)

    def test_transpile_qft_grid(self):
        """Transpile pipeline can handle 8-qubit QFT on 14-qubit grid.
        """
        qr = QuantumRegister(8)
        circuit = QuantumCircuit(qr)
        for i, _ in enumerate(qr):
            for j in range(i):
                circuit.cu1(math.pi / float(2 ** (i - j)), qr[i], qr[j])
            circuit.h(qr[i])

        coupling_map = FakeMelbourne().configuration().coupling_map
        basis_gates = FakeMelbourne().configuration().basis_gates
        new_circuit = transpile(circuit,
                                basis_gates=basis_gates,
                                coupling_map=coupling_map)

        for gate, qargs, _ in new_circuit.data:
            if isinstance(gate, CXGate):
                self.assertIn([x.index for x in qargs], coupling_map)

    def test_already_mapped_1(self):
        """Circuit not remapped if matches topology.

        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """
        backend = FakeRueschlikon()
        coupling_map = backend.configuration().coupling_map
        basis_gates = backend.configuration().basis_gates

        qr = QuantumRegister(16, 'qr')
        cr = ClassicalRegister(16, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[3], qr[14])
        qc.cx(qr[5], qr[4])
        qc.h(qr[9])
        qc.cx(qr[9], qr[8])
        qc.x(qr[11])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[12], qr[11])
        qc.cx(qr[13], qr[4])
        qc.measure(qr, cr)

        new_qc = transpile(qc, coupling_map=coupling_map, basis_gates=basis_gates,
                           initial_layout=Layout.generate_trivial_layout(qr))
        cx_qubits = [qargs for (gate, qargs, _) in new_qc.data if gate.name == "cx"]
        cx_qubits_physical = [[ctrl.index, tgt.index] for [ctrl, tgt] in cx_qubits]
        self.assertEqual(sorted(cx_qubits_physical),
                         [[3, 4], [3, 14], [5, 4], [9, 8], [12, 11], [13, 4]])

    def test_already_mapped_via_layout(self):
        """Test that a manual layout that satisfies a coupling map does not get altered.

        See: https://github.com/Qiskit/qiskit-terra/issues/2036
        """
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']
        coupling_map = [[0, 1], [0, 5], [1, 0], [1, 2], [2, 1], [2, 3],
                        [3, 2], [3, 4], [4, 3], [4, 9], [5, 0], [5, 6],
                        [5, 10], [6, 5], [6, 7], [7, 6], [7, 8], [7, 12],
                        [8, 7], [8, 9], [9, 4], [9, 8], [9, 14], [10, 5],
                        [10, 11], [10, 15], [11, 10], [11, 12], [12, 7],
                        [12, 11], [12, 13], [13, 12], [13, 14], [14, 9],
                        [14, 13], [14, 19], [15, 10], [15, 16], [16, 15],
                        [16, 17], [17, 16], [17, 18], [18, 17], [18, 19],
                        [19, 14], [19, 18]]

        q = QuantumRegister(6, name='qn')
        c = ClassicalRegister(2, name='cn')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[5])
        qc.cx(q[0], q[5])
        qc.u1(2, q[5])
        qc.cx(q[0], q[5])
        qc.h(q[0])
        qc.h(q[5])
        qc.barrier(q)
        qc.measure(q[0], c[0])
        qc.measure(q[5], c[1])

        initial_layout = [q[3], q[4], None, None, q[5], q[2], q[1], None, None, q[0],
                          None, None, None, None, None, None, None, None, None, None]

        new_qc = transpile(qc, coupling_map=coupling_map,
                           basis_gates=basis_gates, initial_layout=initial_layout)
        cx_qubits = [qargs for (gate, qargs, _) in new_qc.data
                     if gate.name == "cx"]
        cx_qubits_physical = [[ctrl.index, tgt.index] for [ctrl, tgt] in cx_qubits]
        self.assertEqual(sorted(cx_qubits_physical),
                         [[9, 4], [9, 4]])

    def test_transpile_bell(self):
        """Test Transpile Bell.

        If all correct some should exists.
        """
        backend = BasicAer.get_backend('qasm_simulator')

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuits = transpile(qc, backend)
        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpile_two(self):
        """Test transpile to circuits.

        If all correct some should exists.
        """
        backend = BasicAer.get_backend('qasm_simulator')

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qubit_reg2 = QuantumRegister(2)
        clbit_reg2 = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, qubit_reg2, clbit_reg, clbit_reg2, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        circuits = transpile([qc, qc_extra], backend)
        self.assertIsInstance(circuits[0], QuantumCircuit)
        self.assertIsInstance(circuits[1], QuantumCircuit)

    def test_mapping_correction(self):
        """Test mapping works in previous failed case.
        """
        backend = FakeRueschlikon()
        qr = QuantumRegister(name='qr', size=11)
        cr = ClassicalRegister(name='qc', size=11)
        circuit = QuantumCircuit(qr, cr)
        circuit.u3(1.564784764685993, -1.2378965763410095, 2.9746763177861713, qr[3])
        circuit.u3(1.2269835563676523, 1.1932982847014162, -1.5597357740824318, qr[5])
        circuit.cx(qr[5], qr[3])
        circuit.u1(0.856768317675967, qr[3])
        circuit.u3(-3.3911273825190915, 0.0, 0.0, qr[5])
        circuit.cx(qr[3], qr[5])
        circuit.u3(2.159209321625547, 0.0, 0.0, qr[5])
        circuit.cx(qr[5], qr[3])
        circuit.u3(0.30949966910232335, 1.1706201763833217, 1.738408691990081, qr[3])
        circuit.u3(1.9630571407274755, -0.6818742967975088, 1.8336534616728195, qr[5])
        circuit.u3(1.330181833806101, 0.6003162754946363, -3.181264980452862, qr[7])
        circuit.u3(0.4885914820775024, 3.133297443244865, -2.794457469189904, qr[8])
        circuit.cx(qr[8], qr[7])
        circuit.u1(2.2196187596178616, qr[7])
        circuit.u3(-3.152367609631023, 0.0, 0.0, qr[8])
        circuit.cx(qr[7], qr[8])
        circuit.u3(1.2646005789809263, 0.0, 0.0, qr[8])
        circuit.cx(qr[8], qr[7])
        circuit.u3(0.7517780502091939, 1.2828514296564781, 1.6781179605443775, qr[7])
        circuit.u3(0.9267400575390405, 2.0526277839695153, 2.034202361069533, qr[8])
        circuit.u3(2.550304293455634, 3.8250017126569698, -2.1351609599720054, qr[1])
        circuit.u3(0.9566260876600556, -1.1147561503064538, 2.0571590492298797, qr[4])
        circuit.cx(qr[4], qr[1])
        circuit.u1(2.1899329069137394, qr[1])
        circuit.u3(-1.8371715243173294, 0.0, 0.0, qr[4])
        circuit.cx(qr[1], qr[4])
        circuit.u3(0.4717053496327104, 0.0, 0.0, qr[4])
        circuit.cx(qr[4], qr[1])
        circuit.u3(2.3167620677708145, -1.2337330260253256, -0.5671322899563955, qr[1])
        circuit.u3(1.0468499525240678, 0.8680750644809365, -1.4083720073192485, qr[4])
        circuit.u3(2.4204244021892807, -2.211701932616922, 3.8297006565735883, qr[10])
        circuit.u3(0.36660280497727255, 3.273119149343493, -1.8003362351299388, qr[6])
        circuit.cx(qr[6], qr[10])
        circuit.u1(1.067395863586385, qr[10])
        circuit.u3(-0.7044917541291232, 0.0, 0.0, qr[6])
        circuit.cx(qr[10], qr[6])
        circuit.u3(2.1830003849921527, 0.0, 0.0, qr[6])
        circuit.cx(qr[6], qr[10])
        circuit.u3(2.1538343756723917, 2.2653381826084606, -3.550087952059485, qr[10])
        circuit.u3(1.307627685019188, -0.44686656993522567, -2.3238098554327418, qr[6])
        circuit.u3(2.2046797998462906, 0.9732961754855436, 1.8527865921467421, qr[9])
        circuit.u3(2.1665254613904126, -1.281337664694577, -1.2424905413631209, qr[0])
        circuit.cx(qr[0], qr[9])
        circuit.u1(2.6209599970201007, qr[9])
        circuit.u3(0.04680566321901303, 0.0, 0.0, qr[0])
        circuit.cx(qr[9], qr[0])
        circuit.u3(1.7728411151289603, 0.0, 0.0, qr[0])
        circuit.cx(qr[0], qr[9])
        circuit.u3(2.4866395967434443, 0.48684511243566697, -3.0069186877854728, qr[9])
        circuit.u3(1.7369112924273789, -4.239660866163805, 1.0623389015296005, qr[0])
        circuit.barrier(qr)
        circuit.measure(qr, cr)

        circuits = transpile(circuit, backend)

        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpiler_layout_from_intlist(self):
        """A list of ints gives layout to correctly map circuit.
        virtual  physical
         q1_0  -  4   ---[H]---
         q2_0  -  5
         q2_1  -  6   ---[H]---
         q3_0  -  8
         q3_1  -  9
         q3_2  -  10  ---[H]---

        """
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        qr3 = QuantumRegister(3, 'qr3')
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc.h(qr1[0])
        qc.h(qr2[1])
        qc.h(qr3[2])
        layout = [4, 5, 6, 8, 9, 10]

        cmap = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10],
                [5, 4], [5, 6], [5, 9], [6, 8], [7, 8],
                [9, 8], [9, 10], [11, 3], [11, 10],
                [11, 12], [12, 2], [13, 1], [13, 12]]

        new_circ = transpile(qc, backend=None,
                             coupling_map=cmap,
                             basis_gates=['u2'],
                             initial_layout=layout)
        mapped_qubits = []
        for _, qargs, _ in new_circ.data:
            mapped_qubits.append(qargs[0].index)

        self.assertEqual(mapped_qubits, [4, 6, 10])

    def test_mapping_multi_qreg(self):
        """Test mapping works for multiple qregs.
        """
        backend = FakeRueschlikon()
        qr = QuantumRegister(3, name='qr')
        qr2 = QuantumRegister(1, name='qr2')
        qr3 = QuantumRegister(4, name='qr3')
        cr = ClassicalRegister(3, name='cr')
        qc = QuantumCircuit(qr, qr2, qr3, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr2[0])
        qc.cx(qr[1], qr3[2])
        qc.measure(qr, cr)

        circuits = transpile(qc, backend)

        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpile_circuits_diff_registers(self):
        """Transpile list of circuits with different qreg names.
        """
        backend = FakeRueschlikon()
        circuits = []
        for _ in range(2):
            qr = QuantumRegister(2)
            cr = ClassicalRegister(2)
            circuit = QuantumCircuit(qr, cr)
            circuit.h(qr[0])
            circuit.cx(qr[0], qr[1])
            circuit.measure(qr, cr)
            circuits.append(circuit)

        circuits = transpile(circuits, backend)
        self.assertIsInstance(circuits[0], QuantumCircuit)

    def test_wrong_initial_layout(self):
        """Test transpile with a bad initial layout.
        """
        backend = FakeMelbourne()

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        bad_initial_layout = [QuantumRegister(3, 'q')[0],
                              QuantumRegister(3, 'q')[1],
                              QuantumRegister(3, 'q')[2]]

        self.assertRaises(DAGCircuitError, transpile,
                          qc, backend, initial_layout=bad_initial_layout)

    def test_parameterized_circuit_for_simulator(self):
        """Verify that a parameterized circuit can be transpiled for a simulator backend."""
        qr = QuantumRegister(2, name='qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        qc.rz(theta, qr[0])

        transpiled_qc = transpile(qc, backend=BasicAer.get_backend('qasm_simulator'))

        expected_qc = QuantumCircuit(qr)
        expected_qc.u1(theta, qr[0])

        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameterized_circuit_for_device(self):
        """Verify that a parameterized circuit can be transpiled for a device backend."""
        qr = QuantumRegister(2, name='qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        qc.rz(theta, qr[0])

        transpiled_qc = transpile(qc, backend=FakeMelbourne(),
                                  initial_layout=Layout.generate_trivial_layout(qr))

        qr = QuantumRegister(14, 'q')
        expected_qc = QuantumCircuit(qr)
        expected_qc.u1(theta, qr[0])

        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameter_expression_circuit_for_simulator(self):
        """Verify that a circuit including expressions of parameters can be
        transpiled for a simulator backend."""
        qr = QuantumRegister(2, name='qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        square = theta * theta
        qc.rz(square, qr[0])

        transpiled_qc = transpile(qc, backend=BasicAer.get_backend('qasm_simulator'))

        expected_qc = QuantumCircuit(qr)
        expected_qc.u1(square, qr[0])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameter_expression_circuit_for_device(self):
        """Verify that a circuit including expressions of parameters can be
        transpiled for a device backend."""
        qr = QuantumRegister(2, name='qr')
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        square = theta * theta
        qc.rz(square, qr[0])

        transpiled_qc = transpile(qc, backend=FakeMelbourne(),
                                  initial_layout=Layout.generate_trivial_layout(qr))

        qr = QuantumRegister(14, 'q')
        expected_qc = QuantumCircuit(qr)
        expected_qc.u1(square, qr[0])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_final_measurement_barrier_for_devices(self):
        """Verify BarrierBeforeFinalMeasurements pass is called in default pipeline for devices."""

        circ = QuantumCircuit.from_qasm_file(self._get_resource_path('example.qasm', Path.QASMS))
        layout = Layout.generate_trivial_layout(*circ.qregs)
        orig_pass = BarrierBeforeFinalMeasurements()
        with patch.object(BarrierBeforeFinalMeasurements, 'run', wraps=orig_pass.run) as mock_pass:
            transpile(circ, coupling_map=FakeRueschlikon().configuration().coupling_map,
                      initial_layout=layout)
            self.assertTrue(mock_pass.called)

    def test_do_not_run_cxdirection_with_symmetric_cm(self):
        """When the coupling map is symmetric, do not run CXDirection."""

        circ = QuantumCircuit.from_qasm_file(self._get_resource_path('example.qasm', Path.QASMS))
        layout = Layout.generate_trivial_layout(*circ.qregs)
        coupling_map = []
        for node1, node2 in FakeRueschlikon().configuration().coupling_map:
            coupling_map.append([node1, node2])
            coupling_map.append([node2, node1])

        orig_pass = CXDirection(CouplingMap(coupling_map))
        with patch.object(CXDirection, 'run', wraps=orig_pass.run) as mock_pass:
            transpile(circ, coupling_map=coupling_map, initial_layout=layout)
            self.assertFalse(mock_pass.called)

    def test_optimize_to_nothing(self):
        """ Optimize gates up to fixed point in the default pipeline
        See https://github.com/Qiskit/qiskit-terra/issues/2035 """
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[0])
        circ.y(qr[0])
        circ.z(qr[0])
        circ.cx(qr[0], qr[1])
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[0], qr[1])

        after = transpile(circ, coupling_map=[[0, 1], [1, 0]],
                          basis_gates=['u3', 'cx'])

        expected = QuantumCircuit(QuantumRegister(2, 'q'))
        self.assertEqual(after, expected)

    def test_pass_manager_empty(self):
        """Test passing an empty PassManager() to the transpiler.

        It should perform no transformations on the circuit.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        resources_before = circuit.count_ops()

        pass_manager = PassManager()
        out_circuit = pass_manager.run(circuit)
        resources_after = out_circuit.count_ops()

        self.assertDictEqual(resources_before, resources_after)

    def test_move_measurements(self):
        """Measurements applied AFTER swap mapping.
        """
        backend = FakeRueschlikon()
        cmap = backend.configuration().coupling_map
        circ = QuantumCircuit.from_qasm_file(
            self._get_resource_path('move_measurements.qasm', Path.QASMS))

        lay = [0, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6]
        out = transpile(circ, initial_layout=lay, coupling_map=cmap)
        out_dag = circuit_to_dag(out)
        meas_nodes = out_dag.named_nodes('measure')
        for meas_node in meas_nodes:
            is_last_measure = all([after_measure.type == 'out'
                                   for after_measure in out_dag.quantum_successors(meas_node)])
            self.assertTrue(is_last_measure)

    def test_initialize_reset_should_be_removed(self):
        """The reset in front of initializer should be removed when zero state"""
        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], [qr[0]])
        qc.initialize([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], [qr[0]])

        expected = QuantumCircuit(qr)
        expected.u3(1.5708, 0, 0, qr[0])
        expected.reset(qr[0])
        expected.u3(1.5708, 3.1416, 0, qr[0])

        after = transpile(qc, basis_gates=['reset', 'u3'], optimization_level=1)
        self.assertEqual(after, expected)

    def test_initialize_FakeMelbourne(self):
        """Test that the zero-state resets are remove in a device not supporting them.
        """
        desired_vector = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])

        out = transpile(qc, backend=FakeMelbourne())
        out_dag = circuit_to_dag(out)
        reset_nodes = out_dag.named_nodes('reset')

        self.assertEqual(reset_nodes, [])

    def test_non_standard_basis(self):
        """Test a transpilation with a non-standard basis"""
        qr1 = QuantumRegister(1, 'q1')
        qr2 = QuantumRegister(2, 'q2')
        qr3 = QuantumRegister(3, 'q3')
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc.h(qr1[0])
        qc.h(qr2[1])
        qc.h(qr3[2])
        layout = [4, 5, 6, 8, 9, 10]

        cmap = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4], [5, 6], [5, 9],
                [6, 8], [7, 8], [9, 8], [9, 10], [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                [13, 12]]

        circuit = transpile(qc, backend=None, coupling_map=cmap,
                            basis_gates=['h'], initial_layout=layout)

        dag_circuit = circuit_to_dag(circuit)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({'h': 3}, resources_after)

    def test_basis_subset(self):
        """Test a transpilation with a basis subset of the standard basis"""
        qr = QuantumRegister(1, 'q1')
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.x(qr[0])
        qc.t(qr[0])

        layout = [4, 5, 6, 8, 9, 10]

        cmap = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4], [5, 6], [5, 9],
                [6, 8], [7, 8], [9, 8], [9, 10], [11, 3], [11, 10], [11, 12], [12, 2], [13, 1],
                [13, 12]]

        circuit = transpile(qc, backend=None, coupling_map=cmap,
                            basis_gates=['u3'], initial_layout=layout)

        dag_circuit = circuit_to_dag(circuit)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({'u3': 1}, resources_after)

    def test_check_circuit_width(self):
        """Verify transpilation of circuit with virtual qubits greater than
        physical qubits raises error"""
        cmap = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4],
                [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10],
                [11, 3], [11, 10], [11, 12], [12, 2], [13, 1], [13, 12]]

        qc = QuantumCircuit(15, 15)

        with self.assertRaises(TranspilerError):
            transpile(qc, coupling_map=cmap)

    @data(0, 1, 2, 3)
    def test_ms_unrolls_to_cx(self, optimization_level):
        """Verify a Rx,Ry,Rxx circuit transpile to a U3,CX target."""

        qc = QuantumCircuit(2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)

        out = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=optimization_level)

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_ms_can_target_ms(self, optimization_level):
        """Verify a Rx,Ry,Rxx circuit can transpile to an Rx,Ry,Rxx target."""

        qc = QuantumCircuit(2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)

        out = transpile(qc, basis_gates=['rx', 'ry', 'rxx'], optimization_level=optimization_level)

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_cx_can_target_ms(self, optimization_level):
        """Verify a U3,CX circuit can transpiler to a Rx,Ry,Rxx target."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(math.pi / 4, [0, 1])

        out = transpile(qc, basis_gates=['rx', 'ry', 'rxx'], optimization_level=optimization_level)

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_measure_doesnt_unroll_ms(self, optimization_level):
        """Verify a measure doesn't cause an Rx,Ry,Rxx circuit to unroll to U3,CX."""

        qc = QuantumCircuit(2, 2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)
        qc.measure([0, 1], [0, 1])

        out = transpile(qc, basis_gates=['rx', 'ry', 'rxx'], optimization_level=optimization_level)

        self.assertEqual(qc, out)

    @data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
    )
    def test_block_collection_runs_for_non_cx_bases(self, basis_gates):
        """Verify block collection is run when a single two qubit gate is in the basis."""
        twoq_gate, *_ = basis_gates

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 1)

        out = transpile(qc, basis_gates=basis_gates, optimization_level=3)

        self.assertLessEqual(out.count_ops()[twoq_gate], 2)

    @unpack
    @data(
        (['u3', 'cx'], {'u3': 1, 'cx': 1}),
        (['rx', 'rz', 'iswap'], {'rx': 6, 'rz': 12, 'iswap': 2}),
        (['rx', 'ry', 'rxx'], {'rx': 6, 'ry': 5, 'rxx': 1}),
    )
    def test_block_collection_reduces_1q_gate(self, basis_gates, gate_counts):
        """For synthesis to non-U3 bases, verify we minimize 1q gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        out = transpile(qc, basis_gates=basis_gates, optimization_level=3)

        self.assertTrue(Operator(out).equiv(qc))
        for basis_gate in basis_gates:
            self.assertLessEqual(out.count_ops()[basis_gate], gate_counts[basis_gate])


class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        raise sys.exc_info()


class TestLogTranspile(QiskitTestCase):
    """Testing the log_transpile option."""

    def setUp(self):
        logger = getLogger()
        logger.setLevel('DEBUG')
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertTranspileLog(self, log_msg):
        """ Runs the transpiler and check for logs containing specified message"""
        transpile(self.circuit)
        self.output.seek(0)
        # Filter unrelated log lines
        output_lines = self.output.readlines()
        transpile_log_lines = [x for x in output_lines if log_msg in x]
        self.assertTrue(len(transpile_log_lines) > 0)

    def test_transpile_log_time(self):
        """Check Total Transpile Time is logged"""
        self.assertTranspileLog('Total Transpile Time')


class TestTranspileCustomPM(QiskitTestCase):
    """Test transpile function with custom pass manager"""

    def test_custom_multiple_circuits(self):
        """Test transpiling with custom pass manager and multiple circuits.
        This tests created a deadlock, so it needs to be monitored for timeout.
        See: https://github.com/Qiskit/qiskit-terra/issues/3925
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pm_conf = PassManagerConfig(
            initial_layout=None,
            basis_gates=['u1', 'u2', 'u3', 'cx'],
            coupling_map=CouplingMap([[0, 1]]),
            backend_properties=None,
            seed_transpiler=1
        )
        passmanager = level_0_pass_manager(pm_conf)

        transpiled = passmanager.run([qc, qc])

        expected = QuantumCircuit(QuantumRegister(2, 'q'))
        expected.u2(0, 3.141592653589793, 0)
        expected.cx(0, 1)

        self.assertEqual(len(transpiled), 2)
        self.assertEqual(transpiled[0], expected)
        self.assertEqual(transpiled[1], expected)
