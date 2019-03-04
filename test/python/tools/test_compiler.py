# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Compiler Test."""

import unittest
from unittest.mock import patch

from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import PassManager, transpile, transpile_dag
from qiskit import compile, execute
from qiskit.test import QiskitTestCase, Path
from qiskit.test.mock import FakeRueschlikon, FakeTenerife
from qiskit.qobj import Qobj
from qiskit.converters import circuit_to_dag
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit.mapper.compiling import two_qubit_kak
from qiskit.mapper.mapping import MapperError
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements


class TestCompiler(QiskitTestCase):
    """Qiskit Compiler Tests."""

    def setUp(self):
        self.seed = 42
        self.backend = BasicAer.get_backend("qasm_simulator")

    def test_compile(self):
        """Test Compiler.

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

    def test_compile_two(self):
        """Test Compiler.

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

    def test_mapping_already_satisfied(self):
        """Test compiler doesn't change circuit already matching backend coupling
        """
        backend = FakeRueschlikon()
        qr = QuantumRegister(16)
        cr = ClassicalRegister(16)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[1])
        qc.x(qr[2])
        qc.x(qr[3])
        qc.x(qr[4])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[3], qr[14])
        qc.measure(qr, cr)
        qobj = compile(qc, backend)
        compiled_ops = qobj.experiments[0].instructions
        original_cx_qubits = [[1, 2], [2, 3], [3, 4], [3, 14]]
        for operation in compiled_ops:
            if operation.name == 'cx':
                self.assertIn(operation.qubits, backend.configuration().coupling_map)
                self.assertIn(operation.qubits, original_cx_qubits)

    def test_compile_circuits_diff_registers(self):
        """Compile list of circuits with different qreg names.
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

    def test_example_multiple_compile(self):
        """Test a toy example compiling multiple circuits.

        Pass if the results are correct.
        """
        backend = BasicAer.get_backend('qasm_simulator')
        coupling_map = [[0, 1], [0, 2],
                        [1, 2],
                        [3, 2], [3, 4],
                        [4, 2]]

        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        bell = QuantumCircuit(qr, cr)
        ghz = QuantumCircuit(qr, cr)
        # Create a GHZ state
        ghz.h(qr[0])
        for i in range(4):
            ghz.cx(qr[i], qr[i + 1])
        # Insert a barrier before measurement
        ghz.barrier()
        # Measure all of the qubits in the standard basis
        for i in range(5):
            ghz.measure(qr[i], cr[i])
        # Create a Bell state
        bell.h(qr[0])
        bell.cx(qr[0], qr[1])
        bell.barrier()
        bell.measure(qr[0], cr[0])
        bell.measure(qr[1], cr[1])
        shots = 2048
        bell_qobj = compile(bell, backend=backend,
                            shots=shots, seed=10)
        ghz_qobj = compile(ghz, backend=backend,
                           shots=shots, coupling_map=coupling_map,
                           seed=10)
        bell_result = backend.run(bell_qobj).result()
        ghz_result = backend.run(ghz_qobj).result()

        threshold = 0.05 * shots
        counts_bell = bell_result.get_counts()
        target_bell = {'00000': shots / 2, '00011': shots / 2}
        self.assertDictAlmostEqual(counts_bell, target_bell, threshold)

        counts_ghz = ghz_result.get_counts()
        target_ghz = {'00000': shots / 2, '11111': shots / 2}
        self.assertDictAlmostEqual(counts_ghz, target_ghz, threshold)

    def test_compile_coupling_map(self):
        """Test compile_coupling_map.
        If all correct should return data with the same stats. The circuit may
        be different.
        """
        backend = BasicAer.get_backend('qasm_simulator')

        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        shots = 2048
        coupling_map = [[0, 1], [1, 2]]
        # TODO (luciano): this initial_layout should be replaced by
        #  {(qr, 0): 0, (qr, 1): 1, (qr, 2): 2} after 0.8
        initial_layout = {("qr", 0): ("q", 0), ("qr", 1): ("q", 1),
                          ("qr", 2): ("q", 2)}
        qobj = compile(qc, backend=backend, shots=shots,
                       coupling_map=coupling_map,
                       initial_layout=initial_layout, seed=88)
        job = backend.run(qobj)
        result = job.result()
        qasm_to_check = qc.qasm()
        self.assertEqual(len(qasm_to_check), 173)

        counts = result.get_counts(qc)
        target = {'000': shots / 2, '111': shots / 2}
        threshold = 0.05 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_example_swap_bits(self):
        """Test a toy example swapping a set bit around.

        Uses the mapper. Pass if results are correct.
        """
        backend = BasicAer.get_backend('qasm_simulator')
        coupling_map = [[0, 1], [0, 8], [1, 2], [1, 9], [2, 3], [2, 10],
                        [3, 4], [3, 11], [4, 5], [4, 12], [5, 6], [5, 13],
                        [6, 7], [6, 14], [7, 15], [8, 9], [9, 10], [10, 11],
                        [11, 12], [12, 13], [13, 14], [14, 15]]

        n = 3  # make this at least 3
        qr0 = QuantumRegister(n)
        qr1 = QuantumRegister(n)
        ans = ClassicalRegister(2 * n)
        qc = QuantumCircuit(qr0, qr1, ans)
        # Set the first bit of qr0
        qc.x(qr0[0])
        # Swap the set bit
        qc.swap(qr0[0], qr0[n - 1])
        qc.swap(qr0[n - 1], qr1[n - 1])
        qc.swap(qr1[n - 1], qr0[1])
        qc.swap(qr0[1], qr1[1])
        # Insert a barrier before measurement
        qc.barrier()
        # Measure all of the qubits in the standard basis
        for j in range(n):
            qc.measure(qr0[j], ans[j])
            qc.measure(qr1[j], ans[j + n])
        # First version: no mapping
        result = execute(qc, backend=backend,
                         coupling_map=None, shots=1024,
                         seed=14).result()
        self.assertEqual(result.get_counts(qc), {'010000': 1024})
        # Second version: map to coupling graph
        result = execute(qc, backend=backend,
                         coupling_map=coupling_map, shots=1024,
                         seed=14).result()
        self.assertEqual(result.get_counts(qc), {'010000': 1024})

    def test_parallel_compile(self):
        """Trigger parallel routines in compile.
        """
        backend = FakeRueschlikon()
        qr = QuantumRegister(16)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        for k in range(1, 15):
            qc.cx(qr[0], qr[k])
        qc.measure(qr[5], cr[0])
        qlist = [qc for k in range(10)]
        qobj = compile(qlist, backend=backend)
        self.assertEqual(len(qobj.experiments), 10)

    def test_compile_pass_manager(self):
        """Test compile with and without an empty pass manager."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.barrier(qr)
        qc.measure(qr, cr)
        backend = BasicAer.get_backend('qasm_simulator')
        qrtrue = compile(qc, backend, seed=42)
        rtrue = backend.run(qrtrue).result()
        qrfalse = compile(qc, backend, seed=42, pass_manager=PassManager())
        rfalse = backend.run(qrfalse).result()
        self.assertEqual(rtrue.get_counts(), rfalse.get_counts())

    def test_compile_with_initial_layout(self):
        """Test compile with an initial layout.
        Regression test for #1711
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])
        initial_layout = {0: (qr, 1), 2: (qr, 0), 15: (qr, 2)}
        backend = FakeRueschlikon()

        qobj = compile(qc, backend, seed=42, initial_layout=initial_layout)

        compiled_ops = qobj.experiments[0].instructions
        for operation in compiled_ops:
            if operation.name == 'cx':
                self.assertIn(operation.qubits, backend.configuration().coupling_map)
                self.assertIn(operation.qubits, [[15, 0], [15, 2]])

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
        self.assertDictAlmostEqual(count1, count2, shots * 0.02)

    def test_grovers_circuit(self):
        """Testing a circuit originated in the Grover algorithm"""
        shots = 1000
        coupling_map = None

        # 6-qubit grovers
        qr = QuantumRegister(6)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr, name='grovers')

        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.x(qr[2])
        circuit.x(qr[3])
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.x(qr[0])
        circuit.cx(qr[1], qr[3])
        circuit.ccx(qr[2], qr[3], qr[4])
        circuit.cx(qr[1], qr[3])
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.x(qr[0])
        circuit.x(qr[1])
        circuit.x(qr[4])
        circuit.h(qr[4])
        circuit.ccx(qr[0], qr[1], qr[4])
        circuit.h(qr[4])
        circuit.x(qr[0])
        circuit.x(qr[1])
        circuit.x(qr[4])
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.h(qr[4])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])

        result = execute(circuit, backend=self.backend,
                         coupling_map=coupling_map, seed=self.seed, shots=shots)
        counts = result.result().get_counts()

        expected_probs = {'00': 0.64,
                          '01': 0.117,
                          '10': 0.113,
                          '11': 0.13}

        target = {key: shots * val for key, val in expected_probs.items()}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

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
        backend = FakeRueschlikon()
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
        backend = FakeTenerife()
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
        backend = FakeRueschlikon()
        cmap = backend.configuration().coupling_map
        circ = QuantumCircuit.from_qasm_file(
            self._get_resource_path('move_measurements.qasm', Path.QASMS))

        dag_circuit = circuit_to_dag(circ)
        lay = {('qa', 0): ('q', 0), ('qa', 1): ('q', 1), ('qb', 0): ('q', 15),
               ('qb', 1): ('q', 2), ('qb', 2): ('q', 14), ('qN', 0): ('q', 3),
               ('qN', 1): ('q', 13), ('qN', 2): ('q', 4), ('qc', 0): ('q', 12),
               ('qNt', 0): ('q', 5), ('qNt', 1): ('q', 11), ('qt', 0): ('q', 6)}
        out_dag = transpile_dag(dag_circuit, initial_layout=lay,
                                coupling_map=cmap)
        meas_nodes = out_dag.named_nodes('measure')
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

    barrier_pass = BarrierBeforeFinalMeasurements()

    @patch.object(BarrierBeforeFinalMeasurements, 'run', wraps=barrier_pass.run)
    def test_final_measurement_barrier_for_devices(self, mock_pass):
        """Verify BarrierBeforeFinalMeasurements pass is called in default pipeline for devices."""

        circ = QuantumCircuit.from_qasm_file(self._get_resource_path('example.qasm', Path.QASMS))
        dag_circuit = circuit_to_dag(circ)
        transpile_dag(dag_circuit, coupling_map=FakeRueschlikon().configuration().coupling_map)

        self.assertTrue(mock_pass.called)

    @patch.object(BarrierBeforeFinalMeasurements, 'run', wraps=barrier_pass.run)
    def test_final_measurement_barrier_for_simulators(self, mock_pass):
        """Verify BarrierBeforeFinalMeasurements pass is in default pipeline for simulators."""
        circ = QuantumCircuit.from_qasm_file(self._get_resource_path('example.qasm', Path.QASMS))
        dag_circuit = circuit_to_dag(circ)
        transpile_dag(dag_circuit)

        self.assertTrue(mock_pass.called)


if __name__ == '__main__':
    unittest.main(verbosity=2)
