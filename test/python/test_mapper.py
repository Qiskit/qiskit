# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,redefined-builtin

import unittest

from qiskit import compile, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.models.backendconfiguration import GateConfig
from qiskit.qobj import Qobj
from qiskit.transpiler._transpiler import transpile_dag
from qiskit.mapper._compiling import two_qubit_kak
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit.mapper._mapping import MapperError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase, Path


class FakeQX4BackEnd:
    """A fake QX4 backend."""

    def name(self):
        return 'qiskit_is_cool'

    def configuration(self):
        qx4_cmap = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]

        return BackendConfiguration(
            backend_name='fake_qx4',
            backend_version='0.0.0',
            n_qubits=5,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=qx4_cmap,
        )


class FakeQX5BackEnd:
    """A fake QX5 backend."""

    def name(self):
        return 'qiskit_is_cool'

    def configuration(self):
        qx5_cmap = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                    [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10],
                    [11, 10], [12, 5], [12, 11], [12, 13], [13, 4],
                    [13, 14], [15, 0], [15, 2], [15, 14]]

        return BackendConfiguration(
            backend_name='fake_qx5',
            backend_version='0.0.0',
            n_qubits=16,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=qx5_cmap,
        )


class TestMapper(QiskitTestCase):
    """Test the mapper."""

    def setUp(self):
        self.seed = 42
        self.backend = BasicAer.get_backend("qasm_simulator")

    def test_mapper_overoptimization(self):
        """Check mapper overoptimization.

        The mapper should not change the semantics of the input.
        An overoptimization introduced issue #81:
        https://github.com/Qiskit/qiskit-terra/issues/81
        """
        # -X-.-----
        # -Y-+-S-.-
        # -Z-.-T-+-
        # ---+-H---
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.y(qr[1])
        circ.z(qr[2])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[2], qr[3])
        circ.s(qr[1])
        circ.t(qr[2])
        circ.h(qr[3])
        circ.cx(qr[1], qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])

        coupling_map = [[0, 2], [1, 2], [2, 3]]
        shots = 1000

        result1 = execute(circ, backend=self.backend,
                          coupling_map=coupling_map, seed=self.seed, shots=shots)
        count1 = result1.result().get_counts()
        result2 = execute(circ, backend=self.backend,
                          coupling_map=None, seed=self.seed, shots=shots)
        count2 = result2.result().get_counts()
        self.assertDictAlmostEqual(count1, count2, shots*0.02)

    def test_math_domain_error(self):
        """Check for floating point errors.

        The math library operates over floats and introduces floating point
        errors that should be avoided.
        See: https://github.com/Qiskit/qiskit-terra/issues/111
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.y(qr[0])
        circ.z(qr[2])
        circ.h(qr[2])
        circ.cx(qr[1], qr[0])
        circ.y(qr[2])
        circ.t(qr[2])
        circ.z(qr[2])
        circ.cx(qr[1], qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])

        coupling_map = [[0, 2], [1, 2], [2, 3]]
        shots = 2000
        job = execute(circ, backend=self.backend,
                      coupling_map=coupling_map, seed=self.seed, shots=shots)
        counts = job.result().get_counts()
        target = {'0001': shots / 2, '0101': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_random_parameter_circuit(self):
        """Run a circuit with randomly generated parameters."""
        circ = QuantumCircuit.from_qasm_file(
            self._get_resource_path('random_n5_d5.qasm', Path.QASMS))
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]
        shots = 1024
        qobj = execute(circ, backend=self.backend,
                       coupling_map=coupling_map, shots=shots, seed=self.seed)
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

    def test_already_mapped(self):
        """Circuit not remapped if matches topology.

        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """
        backend = FakeQX5BackEnd()
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
        for j in range(16):
            qc.measure(qr[j], cr[j])
        qobj = compile(qc, backend=backend)
        cx_qubits = [x.qubits
                     for x in qobj.experiments[0].instructions
                     if x.name == "cx"]

        self.assertEqual(sorted(cx_qubits), [[3, 4], [3, 14], [5, 4],
                                             [9, 8], [12, 11], [13, 4]])

    def test_yzy_zyz_cases(self):
        """yzy_to_zyz works in previously failed cases.

        See: https://github.com/Qiskit/qiskit-terra/issues/607
        """
        backend = FakeQX4BackEnd()
        qr = QuantumRegister(2)
        circ1 = QuantumCircuit(qr)
        circ1.cx(qr[0], qr[1])
        circ1.rz(0.7, qr[1])
        circ1.rx(1.570796, qr[1])
        qobj1 = compile(circ1, backend)
        self.assertIsInstance(qobj1, Qobj)

        circ2 = QuantumCircuit(qr)
        circ2.y(qr[0])
        circ2.h(qr[0])
        circ2.s(qr[0])
        circ2.h(qr[0])
        qobj2 = compile(circ2, backend)
        self.assertIsInstance(qobj2, Qobj)

    def test_move_measurements(self):
        """Measurements applied AFTER swap mapping.
        """
        backend = FakeQX5BackEnd()
        cmap = backend.configuration().coupling_map
        circ = QuantumCircuit.from_qasm_file(
            self._get_resource_path('move_measurements.qasm', Path.QASMS))

        dag_circuit = circuit_to_dag(circ)
        lay = {('qa', 0): ('q', 0), ('qa', 1): ('q', 1), ('qb', 0): ('q', 15),
               ('qb', 1): ('q', 2), ('qb', 2): ('q', 14), ('qN', 0): ('q', 3),
               ('qN', 1): ('q', 13), ('qN', 2): ('q', 4), ('qc', 0): ('q', 12),
               ('qNt', 0): ('q', 5), ('qNt', 1): ('q', 11), ('qt', 0): ('q', 6)}
        out_dag = transpile_dag(dag_circuit, initial_layout=lay,
                                coupling_map=cmap, format='dag')
        meas_nodes = out_dag.get_named_nodes('measure')
        for n in meas_nodes:
            is_last_measure = all([after_measure in out_dag.output_map.values()
                                   for after_measure in out_dag.quantum_successors(n)])
            self.assertTrue(is_last_measure)

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


if __name__ == '__main__':
    unittest.main()
