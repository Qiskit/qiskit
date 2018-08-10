# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Compiler Test."""

import unittest
import qiskit
from qiskit import transpiler
from qiskit import Result
from qiskit.qobj import Qobj
from qiskit.wrapper import register, available_backends, get_backend, execute, least_busy
from qiskit._qiskiterror import QISKitError
from .common import requires_qe_access, QiskitTestCase


class FakeBackEnd(object):
    """A fake backend.
    """
    def __init__(self):
        qx5_cmap = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                    [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10], [11, 10],
                    [12, 5], [12, 11], [12, 13], [13, 4], [13, 14], [15, 0],
                    [15, 2], [15, 14]]
        self.configuration = {'name': 'fake', 'basis_gates': 'u1,u2,u3,cx,id',
                              'simulator': False, 'n_qubits': 16,
                              'coupling_map': qx5_cmap}


class TestCompiler(QiskitTestCase):
    """QISKit Compiler Tests."""

    def test_compile(self):
        """Test Compiler.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = transpiler.compile(qc, backend)

        # FIXME should validate the Qobj when defined
        self.assertIsInstance(qobj, Qobj)

    def test_compile_two(self):
        """Test Compiler.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = transpiler.compile([qc, qc_extra], backend)

        # FIXME should validate the Qobj when defined
        self.assertIsInstance(qobj, Qobj)

    def test_compile_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = transpiler.compile(qc, backend)
        result = backend.run(qobj).result()
        self.assertIsInstance(result, Result)

    def test_compile_two_run(self):
        """Test Compiler and run.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = transpiler.compile([qc, qc_extra], backend)
        result = backend.run(qobj).result()
        self.assertIsInstance(result, Result)

    def test_execute(self):
        """Test Execute.

        If all correct some should exists.
        """
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
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
        backend = get_backend('local_qasm_simulator')

        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_compile_remote(self, qe_token, qe_url):
        """Test Compiler remote.

        If all correct some should exists.
        """
        register(qe_token, qe_url)
        backend = least_busy(available_backends())
        backend = get_backend(backend)

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = transpiler.compile(qc, backend)

        # FIXME should validate the Qobj when defined
        self.assertIsInstance(qobj, Qobj)

    @requires_qe_access
    def test_compile_two_remote(self, qe_token, qe_url):
        """Test Compiler remote on two circuits.

        If all correct some should exists.
        """
        register(qe_token, qe_url)
        backend = least_busy(available_backends())
        backend = get_backend(backend)

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = transpiler.compile([qc, qc_extra], backend)

        # FIXME should validate the Qobj when defined
        self.assertIsInstance(qobj, Qobj)

    @requires_qe_access
    def test_compile_run_remote(self, qe_token, qe_url):
        """Test Compiler and run remote.

        If all correct some should exists.
        """
        register(qe_token, qe_url)
        backend = available_backends({'local': False, 'simulator': True})[0]
        backend = get_backend(backend)
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qobj = transpiler.compile(qc, backend)
        job = backend.run(qobj)
        result = job.result(timeout=20)
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_compile_two_run_remote(self, qe_token, qe_url):
        """Test Compiler and run two circuits.

        If all correct some should exists.
        """
        register(qe_token, qe_url)
        backend = available_backends({'local': False, 'simulator': True})[0]
        backend = get_backend(backend)
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = transpiler.compile([qc, qc_extra], backend)
        job = backend.run(qobj)
        result = job.result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_execute_remote(self, qe_token, qe_url):
        """Test Execute remote.

        If all correct some should exists.
        """
        register(qe_token, qe_url)
        backend = available_backends({'local': False, 'simulator': True})[0]
        backend = get_backend(backend)
        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        job = execute(qc, backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    @requires_qe_access
    def test_execute_two_remote(self, qe_token, qe_url):
        """Test execute two remote.

        If all correct some should exists.
        """
        register(qe_token, qe_url)
        backend = available_backends({'local': False, 'simulator': True})[0]
        backend = get_backend(backend)
        qubit_reg = qiskit.QuantumRegister(2)
        clbit_reg = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg)
        qc_extra.measure(qubit_reg, clbit_reg)
        job = execute([qc, qc_extra], backend)
        results = job.result()
        self.assertIsInstance(results, Result)

    def test_mapping_correction(self):
        """Test mapping works in previous failed case.
        """
        backend = FakeBackEnd()
        qr = qiskit.QuantumRegister(name='qr', size=11)
        cr = qiskit.ClassicalRegister(name='qc', size=11)
        circuit = qiskit.QuantumCircuit(qr, cr)
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
            qobj = transpiler.compile(circuit, backend)
        except QISKitError:
            qobj = None
        self.assertIsInstance(qobj, Qobj)

    def test_mapping_multi_qreg(self):
        """Test mapping works for multiple qregs.
        """
        backend = FakeBackEnd()
        qr = qiskit.QuantumRegister(3, name='qr')
        qr2 = qiskit.QuantumRegister(1, name='qr2')
        qr3 = qiskit.QuantumRegister(4, name='qr3')
        cr = qiskit.ClassicalRegister(3, name='cr')
        qc = qiskit.QuantumCircuit(qr, qr2, qr3, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr2[0])
        qc.cx(qr[1], qr3[2])
        qc.measure(qr, cr)

        try:
            qobj = transpiler.compile(qc, backend)
        except QISKitError:
            qobj = None
        self.assertIsInstance(qobj, Qobj)

    def test_mapping_already_satisfied(self):
        """Test compiler doesn't change circuit already matching backend coupling
        """
        backend = FakeBackEnd()
        qr = qiskit.QuantumRegister(16)
        cr = qiskit.ClassicalRegister(16)
        qc = qiskit.QuantumCircuit(qr, cr)
        qc.h(qr[1])
        qc.x(qr[2])
        qc.x(qr[3])
        qc.x(qr[4])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[3], qr[14])
        qc.measure(qr, cr)
        qobj = transpiler.compile(qc, backend)
        compiled_ops = qobj.experiments[0].instructions
        for operation in compiled_ops:
            if operation.name == 'cx':
                self.assertIn(operation.qubits, backend.configuration['coupling_map'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
