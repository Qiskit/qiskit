# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin


"""Compiler Test."""

import unittest

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpiler
from qiskit import compile
from qiskit import Result
from qiskit.backends.models import BackendConfiguration
from qiskit.backends.models.backendconfiguration import GateConfig
from qiskit.dagcircuit import DAGCircuit
from qiskit import execute
from qiskit._qiskiterror import QISKitError
from qiskit.backends.ibmq import least_busy
from .common import QiskitTestCase, requires_qe_access, bin_to_hex_keys


class FakeBackend(object):
    """A fake backend.
    """

    def name(self):
        """ name of fake backend"""
        return 'qiskit_is_cool'

    def configuration(self):
        """Return a make up configuration for a fake QX5 device."""
        qx5_cmap = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                    [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10], [11, 10],
                    [12, 5], [12, 11], [12, 13], [13, 4], [13, 14], [15, 0],
                    [15, 2], [15, 14]]
        return BackendConfiguration(
            backend_name='fake',
            backend_version='0.0.0',
            n_qubits=16,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=qx5_cmap,
        )


class TestCompiler(QiskitTestCase):
    """QISKit Compiler Tests."""

    seed = 42

    def test_compile(self):
        """Test Compiler.

        If all correct some should exists.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        dags = transpiler.transpile(qc, backend)
        self.assertIsInstance(dags[0], DAGCircuit)

    def test_compile_two(self):
        """Test Compiler.

        If all correct some should exists.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

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
        dags = transpiler.transpile([qc, qc_extra], backend)
        self.assertIsInstance(dags[0], DAGCircuit)
        self.assertIsInstance(dags[1], DAGCircuit)

    def test_compile_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = compile(qc, backend)
        result = backend.run(qobj).result()
        self.assertIsInstance(result, Result)

    def test_compile_two_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = compile([qc, qc_extra], backend)
        result = backend.run(qobj).result()
        self.assertIsInstance(result, Result)

    def test_execute(self):
        """Test Execute.

        If all correct some should exists.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        job = execute(qc, backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    def test_execute_two(self):
        """Test execute two.

        If all correct some should exists.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_compile_remote(self, qe_token, qe_url):
        """Test Compiler remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = least_busy(qiskit.IBMQ.backends())

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        dags = transpiler.transpile(qc, backend)
        self.assertIsInstance(dags[0], DAGCircuit)

    @requires_qe_access
    def test_compile_two_remote(self, qe_token, qe_url):
        """Test Compiler remote on two circuits.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = least_busy(qiskit.IBMQ.backends())

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        dags = transpiler.transpile([qc, qc_extra], backend)
        self.assertIsInstance(dags[0], DAGCircuit)
        self.assertIsInstance(dags[1], DAGCircuit)

    @requires_qe_access
    def test_compile_run_remote(self, qe_token, qe_url):
        """Test Compiler and run remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qobj = compile(qc, backend, seed=TestCompiler.seed)
        job = backend.run(qobj)
        result = job.result(timeout=20)
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_compile_two_run_remote(self, qe_token, qe_url):
        """Test Compiler and run two circuits.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2, name='q')
        clbit_reg = ClassicalRegister(2, name='c')
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = compile([qc, qc_extra], backend, seed=TestCompiler.seed)
        job = backend.run(qobj)
        result = job.result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_execute_remote(self, qe_token, qe_url):
        """Test Execute remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        job = execute(qc, backend, seed=TestCompiler.seed)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_execute_two_remote(self, qe_token, qe_url):
        """Test execute two remote.

        If all correct some should exists.
        """
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        backend = qiskit.IBMQ.get_backend(local=False, simulator=True)

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend, seed=TestCompiler.seed)
        results = job.result()
        self.assertIsInstance(results, Result)

    def test_mapping_correction(self):
        """Test mapping works in previous failed case.
        """
        backend = FakeBackend()
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

        try:
            dags = transpiler.transpile(circuit, backend)
        except QISKitError:
            dags = None
        self.assertIsInstance(dags[0], DAGCircuit)

    def test_mapping_multi_qreg(self):
        """Test mapping works for multiple qregs.
        """
        backend = FakeBackend()
        qr = QuantumRegister(3, name='qr')
        qr2 = QuantumRegister(1, name='qr2')
        qr3 = QuantumRegister(4, name='qr3')
        cr = ClassicalRegister(3, name='cr')
        qc = QuantumCircuit(qr, qr2, qr3, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr2[0])
        qc.cx(qr[1], qr3[2])
        qc.measure(qr, cr)

        try:
            dags = transpiler.transpile(qc, backend)
        except QISKitError:
            dags = None
        self.assertIsInstance(dags[0], DAGCircuit)

    def test_mapping_already_satisfied(self):
        """Test compiler doesn't change circuit already matching backend coupling
        """
        backend = FakeBackend()
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
        for operation in compiled_ops:
            if operation.name == 'cx':
                self.assertIn(operation.qubits, backend.configuration().coupling_map)

    def test_compile_circuits_diff_registers(self):
        """Compile list of circuits with different qreg names.
        """
        backend = FakeBackend()
        circuits = []
        for _ in range(2):
            qr = QuantumRegister(2)
            cr = ClassicalRegister(2)
            circuit = QuantumCircuit(qr, cr)
            circuit.h(qr[0])
            circuit.cx(qr[0], qr[1])
            circuit.measure(qr, cr)
            circuits.append(circuit)

        dags = transpiler.transpile(circuits, backend)
        self.assertIsInstance(dags[0], DAGCircuit)

    def test_example_multiple_compile(self):
        """Test a toy example compiling multiple circuits.

        Pass if the results are correct.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')
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
            ghz.cx(qr[i], qr[i+1])
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

        threshold = 0.04 * shots
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
        backend = qiskit.Aer.get_backend('qasm_simulator_py')

        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        shots = 1024
        coupling_map = [[0, 1], [1, 2]]
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
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_example_swap_bits(self):
        """Test a toy example swapping a set bit around.

        Uses the mapper. Pass if results are correct.
        """
        backend = qiskit.Aer.get_backend('qasm_simulator_py')
        coupling_map = [[0, 1], [0, 8], [1, 2], [1, 9], [2, 3], [2, 10],
                        [3, 4], [3, 11], [4, 5], [4, 12], [5, 6], [5, 13],
                        [6, 7], [6, 14], [7, 15], [8, 9], [9, 10], [10, 11],
                        [11, 12], [12, 13], [13, 14], [14, 15]]

        n = 3  # make this at least 3
        qr0 = QuantumRegister(n)
        qr1 = QuantumRegister(n)
        ans = ClassicalRegister(2*n)
        qc = QuantumCircuit(qr0, qr1, ans)
        # Set the first bit of qr0
        qc.x(qr0[0])
        # Swap the set bit
        qc.swap(qr0[0], qr0[n-1])
        qc.swap(qr0[n-1], qr1[n-1])
        qc.swap(qr1[n-1], qr0[1])
        qc.swap(qr0[1], qr1[1])
        # Insert a barrier before measurement
        qc.barrier()
        # Measure all of the qubits in the standard basis
        for j in range(n):
            qc.measure(qr0[j], ans[j])
            qc.measure(qr1[j], ans[j+n])
        # First version: no mapping
        result = execute(qc, backend=backend,
                         coupling_map=None, shots=1024,
                         seed=14).result()
        self.assertEqual(result.get_counts(qc),
                         bin_to_hex_keys({'010000': 1024}))
        # Second version: map to coupling graph
        result = execute(qc, backend=backend,
                         coupling_map=coupling_map, shots=1024,
                         seed=14).result()
        self.assertEqual(result.get_counts(qc),
                         bin_to_hex_keys({'010000': 1024}))

    def test_parallel_compile(self):
        """Trigger parallel routines in compile.
        """
        backend = FakeBackend()
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

    def test_already_matching(self):
        """Map qubit i -> i if circuit is already compatible with topology.
        """
        backend = FakeBackend()
        qr = QuantumRegister(16, 'qr')
        cr = ClassicalRegister(4, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        qc.cx(qr[1], qr[0])
        qc.cx(qr[6], qr[11])
        qc.cx(qr[8], qr[7])
        qc.measure(qr[1], cr[0])
        qc.measure(qr[0], cr[1])
        qc.measure(qr[6], cr[2])
        qc.measure(qr[11], cr[3])
        qobj = compile(qc, backend=backend)
        for qubit_layout in qobj.experiments[0].config.layout:
            self.assertEqual(qubit_layout[0][1], qubit_layout[1][1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
