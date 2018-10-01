# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

import unittest

import qiskit.wrapper
from qiskit import load_qasm_string, mapper, qasm, unroll
from qiskit.qobj import Qobj
from qiskit.transpiler._transpiler import transpile
from qiskit.dagcircuit._dagcircuit import DAGCircuit
from qiskit.mapper._compiling import two_qubit_kak
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit.mapper._mapping import remove_last_measurements, MapperError
from .common import QiskitTestCase


class FakeQX4BackEnd(object):
    """A fake QX4 backend.
    """

    def configuration(self):
        qx4_cmap = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
        return {
            'name': 'fake_qx4', 'basis_gates': 'u1,u2,u3,cx,id',
            'simulator': False, 'n_qubits': 5,
            'coupling_map': qx4_cmap
        }


class FakeQX5BackEnd(object):
    """A fake QX5 backend.
    """

    def configuration(self):
        qx5_cmap = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                    [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10],
                    [11, 10], [12, 5], [12, 11], [12, 13], [13, 4],
                    [13, 14], [15, 0], [15, 2], [15, 14]]
        return {
            'name': 'fake_qx5', 'basis_gates': 'u1,u2,u3,cx,id',
            'simulator': False, 'n_qubits': 16,
            'coupling_map': qx5_cmap
        }


class MapperTest(QiskitTestCase):
    """Test the mapper."""

    def setUp(self):
        self.seed = 42

    def test_mapper_overoptimization(self):
        """Check mapper overoptimization

        The mapper should not change the semantics of the input. An overoptimization introduced
        the issue #81: https://github.com/QISKit/qiskit-terra/issues/81
        """
        circ = qiskit.load_qasm_file(self._get_resource_path('qasm/overoptimization.qasm'))
        coupling_map = [[0, 2], [1, 2], [2, 3]]
        result1 = qiskit.execute(circ, backend="qasm_simulator",
                                 coupling_map=coupling_map, seed=self.seed)
        count1 = result1.result().get_counts()
        result2 = qiskit.execute(circ, backend="qasm_simulator", coupling_map=None,
                                 seed=self.seed)
        count2 = result2.result().get_counts()
        self.assertEqual(count1.keys(), count2.keys(), )

    def test_math_domain_error(self):
        """Check for floating point errors.

        The math library operates over floats and introduce floating point errors that should be
        avoided.
        See: https://github.com/QISKit/qiskit-terra/issues/111
        """
        circ = qiskit.load_qasm_file(self._get_resource_path('qasm/math_domain_error.qasm'))
        coupling_map = [[0, 2], [1, 2], [2, 3]]
        shots = 2000
        qobj = qiskit.execute(circ, backend="qasm_simulator",
                              coupling_map=coupling_map,
                              seed=self.seed, shots=shots)
        counts = qobj.result().get_counts()
        target = {'0001': shots / 2, '0101':  shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_optimize_1q_gates_issue159(self):
        """optimize_1q_gates that removes u1(2*pi) rotations.
        See: https://github.com/QISKit/qiskit-terra/issues/159
        """
        qr = qiskit.QuantumRegister(2, 'qr')
        cr = qiskit.ClassicalRegister(2, 'cr')
        qc = qiskit.QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[1], qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'qasm_simulator'
        coupling_map = [[1, 0], [2, 0], [2, 1], [2, 4], [3, 2], [3, 4]]
        initial_layout = {('qr', 0): ('q', 1), ('qr', 1): ('q', 0)}
        qobj = qiskit.compile(qc, backend=backend,
                              initial_layout=initial_layout,
                              coupling_map=coupling_map)

        comp_qasm = qobj.experiments[0].header.compiled_circuit_qasm

        self.assertEqual(comp_qasm, EXPECTED_QASM_1Q_GATES_3_5)

    def test_random_parameter_circuit(self):
        """Run a circuit with randomly generated parameters."""
        circ = qiskit.load_qasm_file(self._get_resource_path('qasm/random_n5_d5.qasm'))
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]
        shots = 1024
        qobj = qiskit.execute(circ, backend="qasm_simulator",
                              coupling_map=coupling_map, shots=shots,
                              seed=self.seed)
        counts = qobj.result().get_counts()
        expected_probs = {
            '00000': 0.079239867254200971,
            '00001': 0.032859032998526903,
            '00010': 0.10752610993531816,
            '00011': 0.018818532050952699,
            '00100': 0.054830807251011054,
            '00101': 0.0034141983951965164,
            '00110': 0.041649309748902276,
            '00111': 0.039967731207338125,
            '01000': 0.10516937819949743,
            '01001': 0.026635620063700002,
            '01010': 0.0053475143548793866,
            '01011': 0.01940513314416064,
            '01100': 0.0044028405481225047,
            '01101': 0.057524760052126644,
            '01110': 0.010795354134597078,
            '01111': 0.026491296821535528,
            '10000': 0.094827455395274859,
            '10001': 0.0008373965072688836,
            '10010': 0.029082297894094441,
            '10011': 0.012386622870598416,
            '10100': 0.018739140061148799,
            '10101': 0.01367656456536896,
            '10110': 0.039184170706009248,
            '10111': 0.062339335178438288,
            '11000': 0.00293674365989009,
            '11001': 0.012848433960739968,
            '11010': 0.018472497159499782,
            '11011': 0.0088903691234912003,
            '11100': 0.031305389080034329,
            '11101': 0.0004788556283690458,
            '11110': 0.002232419390471667,
            '11111': 0.017684822659235985
        }
        target = {key: shots * val for key, val in expected_probs.items()}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_symbolic_unary(self):
        """SymPy with a prefix.

        See: https://github.com/QISKit/qiskit-terra/issues/172
        """
        ast = qasm.Qasm(filename=self._get_resource_path(
            'qasm/issue172_unary.qasm')).parse()
        unr = unroll.Unroller(ast, backend=unroll.DAGBackend(["cx", "u1", "u2", "u3"]))
        unr.execute()
        circ = mapper.optimize_1q_gates(unr.backend.circuit)
        self.assertEqual(circ.qasm(qeflag=True), EXPECTED_QASM_SYMBOLIC_UNARY)

    def test_symbolic_binary(self):
        """SymPy binary operation.

        See: https://github.com/QISKit/qiskit-terra/issues/172
        """
        ast = qasm.Qasm(filename=self._get_resource_path(
            'qasm/issue172_binary.qasm')).parse()

        unr = unroll.Unroller(ast, backend=unroll.DAGBackend(["cx", "u1", "u2", "u3"]))
        unr.execute()
        circ = mapper.optimize_1q_gates(unr.backend.circuit)
        self.assertEqual(circ.qasm(qeflag=True), EXPECTED_QASM_SYMBOLIC_BINARY)

    def test_symbolic_extern(self):
        """SymPy with external function.

        See: https://github.com/QISKit/qiskit-terra/issues/172
        """
        ast = qasm.Qasm(filename=self._get_resource_path(
            'qasm/issue172_extern.qasm')).parse()
        unr = unroll.Unroller(ast, backend=unroll.DAGBackend(["cx", "u1", "u2", "u3"]))
        unr.execute()
        circ = mapper.optimize_1q_gates(unr.backend.circuit)
        self.assertEqual(circ.qasm(qeflag=True), EXPECTED_QASM_SYMBOLIC_EXTERN)

    def test_symbolic_power(self):
        """SymPy with a power (^).

        See: https://github.com/QISKit/qiskit-terra/issues/172
        """
        ast = qasm.Qasm(data=QASM_SYMBOLIC_POWER).parse()
        unr = unroll.Unroller(ast, backend=unroll.DAGBackend(["cx", "u1", "u2", "u3"]))
        unr.execute()
        circ = mapper.optimize_1q_gates(unr.backend.circuit)
        self.assertEqual(circ.qasm(qeflag=True), EXPECTED_QASM_SYMBOLIC_POWER)

    def test_already_mapped(self):
        """Circuit not remapped if matches topology.

        See: https://github.com/QISKit/qiskit-terra/issues/342
        """
        qr = qiskit.QuantumRegister(16, 'qr')
        cr = qiskit.ClassicalRegister(16, 'cr')
        qc = qiskit.QuantumCircuit(qr, cr)
        qc.cx(qr[3], qr[14])
        qc.cx(qr[5], qr[4])
        qc.h(qr[9])
        qc.cx(qr[9], qr[8])
        qc.x(qr[11])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[12], qr[11])
        qc.cx(qr[13], qr[4])
        for j in range(16):
            qc.measure(qr[j], cr[j])
        backend = 'qasm_simulator'
        coupling_map = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4],
                        [6, 5], [6, 7], [6, 11], [7, 10], [8, 7], [9, 8],
                        [9, 10], [11, 10], [12, 5], [12, 11], [12, 13],
                        [13, 4], [13, 14], [15, 0], [15, 2], [15, 14]]
        qobj = qiskit.compile(qc, backend=backend, coupling_map=coupling_map)
        cx_qubits = [x.qubits
                     for x in qobj.experiments[0].instructions
                     if x.name == "cx"]

        self.assertEqual(sorted(cx_qubits), [[3, 4], [3, 14], [5, 4], [9, 8], [12, 11], [13, 4]])

    def test_yzy_zyz_cases(self):
        """yzy_to_zyz works in previously failed cases.

        See: https://github.com/QISKit/qiskit-terra/issues/607
        """
        backend = FakeQX4BackEnd()
        circ1 = load_qasm_string(YZY_ZYZ_1)
        qobj1 = qiskit.wrapper.compile(circ1, backend)
        self.assertIsInstance(qobj1, Qobj)
        circ2 = load_qasm_string(YZY_ZYZ_2)
        qobj2 = qiskit.wrapper.compile(circ2, backend)
        self.assertIsInstance(qobj2, Qobj)

    def test_move_measurements(self):
        """Measurements applied AFTER swap mapping.
        """
        backend = FakeQX5BackEnd()
        cmap = backend.configuration()['coupling_map']
        circ = qiskit.load_qasm_file(self._get_resource_path('qasm/move_measurements.qasm'),
                                     name='move')
        dag_circuit = DAGCircuit.fromQuantumCircuit(circ)
        lay = {('qa', 0): ('q', 0), ('qa', 1): ('q', 1), ('qb', 0): ('q', 15),
               ('qb', 1): ('q', 2), ('qb', 2): ('q', 14), ('qN', 0): ('q', 3),
               ('qN', 1): ('q', 13), ('qN', 2): ('q', 4), ('qc', 0): ('q', 12),
               ('qNt', 0): ('q', 5), ('qNt', 1): ('q', 11), ('qt', 0): ('q', 6)}
        out_dag = transpile(dag_circuit, initial_layout=lay,
                            coupling_map=cmap, format='dag')
        moved_meas = remove_last_measurements(out_dag, perform_remove=False)
        meas_nodes = out_dag.get_named_nodes('measure')
        self.assertEqual(len(moved_meas), len(meas_nodes))

    def test_kak_decomposition(self):
        """Verify KAK decomposition for random Haar unitaries.
        """
        for _ in range(100):
            unitary = random_unitary_matrix(4)
            with self.subTest(unitary=unitary):
                try:
                    two_qubit_kak(unitary, verify_gate_sequence=True)
                except MapperError as ex:
                    self.fail(str(ex))


# QASMs expected by the tests.
EXPECTED_QASM_SYMBOLIC_BINARY = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
u1(-0.1 + 0.55*pi) qr[0];
measure qr[0] -> cr[0];\n"""

EXPECTED_QASM_SYMBOLIC_EXTERN = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
u1(-0.479425538604203) qr[0];
measure qr[0] -> cr[0];\n"""

EXPECTED_QASM_SYMBOLIC_POWER = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
u1(pi + (-pi + 0.3)^2.0) qr[0];
measure qr[0] -> cr[0];\n"""

EXPECTED_QASM_SYMBOLIC_UNARY = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
u1(-1.5*pi) qr[0];
measure qr[0] -> cr[0];\n"""

EXPECTED_QASM_1Q_GATES = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg cr[2];
u2(0,3.14159265358979) q[0];
cx q[1],q[0];
cx q[1],q[0];
cx q[1],q[0];
u2(0,3.14159265358979) q[1];
measure q[1] -> cr[0];
u2(0,3.14159265358979) q[0];
measure q[0] -> cr[1];\n"""

# This QASM is the same as EXPECTED_QASM_1Q_GATES, with the u2-measure lines
# swapped.
EXPECTED_QASM_1Q_GATES_3_5 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg cr[2];
u2(0,3.14159265358979) q[0];
cx q[1],q[0];
cx q[1],q[0];
cx q[1],q[0];
u2(0,3.14159265358979) q[0];
u2(0,3.14159265358979) q[1];
measure q[1] -> cr[0];
measure q[0] -> cr[1];\n"""

QASM_SYMBOLIC_POWER = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
u1(pi) qr[0];
u1((0.3+(-pi))^2) qr[0];
measure qr[0] -> cr[0];"""

YZY_ZYZ_1 = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[2];
cx qr[0],qr[1];
rz(0.7) qr[1];
rx(1.570796) qr[1];
"""

YZY_ZYZ_2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[2];
y qr[0];
h qr[0];
s qr[0];
h qr[0];
"""


if __name__ == '__main__':
    unittest.main()
