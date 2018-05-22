# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Compiler Test."""

import unittest
import qiskit
import qiskit._compiler
from qiskit import Result
from qiskit.wrapper import get_backend, execute
from qiskit.backends.ibmq import IBMQProvider
from qiskit._qiskiterror import QISKitError
from .common import requires_qe_access, QiskitTestCase


def lowest_pending_jobs(list_of_backends):
    """Returns the backend with lowest pending jobs."""
    by_pending_jobs = sorted(list_of_backends,
                             key=lambda x: x.status['pending_jobs'])
    return by_pending_jobs[0]


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

        qobj = qiskit._compiler.compile(qc, backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

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
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

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

        qobj = qiskit._compiler.compile(qc, backend)
        result = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                               preformatted=True)).result()
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
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)
        result = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                               preformatted=True)).result()
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
        job = qiskit.wrapper.execute(qc, backend)
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
    def test_compile_remote(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test Compiler remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = lowest_pending_jobs(
            provider.available_backends({'local': False, 'simulator': False}))

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        qobj = qiskit._compiler.compile(qc, backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    @requires_qe_access
    def test_compile_two_remote(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test Compiler remote on two circuits.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = lowest_pending_jobs(
            provider.available_backends({'local': False, 'simulator': False}))

        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)

        # FIXME should test against the qobj when defined
        self.assertEqual(len(qobj), 3)

    @requires_qe_access
    def test_compile_run_remote(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test Compiler and run remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = provider.available_backends({'simulator': True})[0]
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile(qc, backend)
        result = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                               preformatted=True)).result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_compile_two_run_remote(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test Compiler and run two circuits.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = provider.available_backends({'simulator': True})[0]
        qubit_reg = qiskit.QuantumRegister(2, name='q')
        clbit_reg = qiskit.ClassicalRegister(2, name='c')
        qc = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = qiskit.QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        qobj = qiskit._compiler.compile([qc, qc_extra], backend)
        job = backend.run(qiskit.QuantumJob(qobj, backend=backend,
                                            preformatted=True))
        result = job.result()
        self.assertIsInstance(result, Result)

    @requires_qe_access
    def test_execute_remote(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test Execute remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = provider.available_backends({'simulator': True})[0]
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
    def test_execute_two_remote(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test execute two remote.

        If all correct some should exists.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = provider.available_backends({'simulator': True})[0]
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
    def test_mapping_correction(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test mapping works in previous failed case.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = provider.get_backend('ibmqx5')

        q = qiskit.QuantumRegister(name='qr', size=11)
        c = qiskit.ClassicalRegister(name='qc', size=11)
        circuit = qiskit.QuantumCircuit(q, c)
        circuit.u3(1.564784764685993, -1.2378965763410095, 2.9746763177861713, q[3])
        circuit.u3(1.2269835563676523, 1.1932982847014162, -1.5597357740824318, q[5])
        circuit.cx(q[5], q[3])
        circuit.u1(0.856768317675967, q[3])
        circuit.u3(-3.3911273825190915, 0.0, 0.0, q[5])
        circuit.cx(q[3], q[5])
        circuit.u3(2.159209321625547, 0.0, 0.0, q[5])
        circuit.cx(q[5], q[3])
        circuit.u3(0.30949966910232335, 1.1706201763833217, 1.738408691990081, q[3])
        circuit.u3(1.9630571407274755, -0.6818742967975088, 1.8336534616728195, q[5])
        circuit.u3(1.330181833806101, 0.6003162754946363, -3.181264980452862, q[7])
        circuit.u3(0.4885914820775024, 3.133297443244865, -2.794457469189904, q[8])
        circuit.cx(q[8], q[7])
        circuit.u1(2.2196187596178616, q[7])
        circuit.u3(-3.152367609631023, 0.0, 0.0, q[8])
        circuit.cx(q[7], q[8])
        circuit.u3(1.2646005789809263, 0.0, 0.0, q[8])
        circuit.cx(q[8], q[7])
        circuit.u3(0.7517780502091939, 1.2828514296564781, 1.6781179605443775, q[7])
        circuit.u3(0.9267400575390405, 2.0526277839695153, 2.034202361069533, q[8])
        circuit.u3(2.550304293455634, 3.8250017126569698, -2.1351609599720054, q[1])
        circuit.u3(0.9566260876600556, -1.1147561503064538, 2.0571590492298797, q[4])
        circuit.cx(q[4], q[1])
        circuit.u1(2.1899329069137394, q[1])
        circuit.u3(-1.8371715243173294, 0.0, 0.0, q[4])
        circuit.cx(q[1], q[4])
        circuit.u3(0.4717053496327104, 0.0, 0.0, q[4])
        circuit.cx(q[4], q[1])
        circuit.u3(2.3167620677708145, -1.2337330260253256, -0.5671322899563955, q[1])
        circuit.u3(1.0468499525240678, 0.8680750644809365, -1.4083720073192485, q[4])
        circuit.u3(2.4204244021892807, -2.211701932616922, 3.8297006565735883, q[10])
        circuit.u3(0.36660280497727255, 3.273119149343493, -1.8003362351299388, q[6])
        circuit.cx(q[6], q[10])
        circuit.u1(1.067395863586385, q[10])
        circuit.u3(-0.7044917541291232, 0.0, 0.0, q[6])
        circuit.cx(q[10], q[6])
        circuit.u3(2.1830003849921527, 0.0, 0.0, q[6])
        circuit.cx(q[6], q[10])
        circuit.u3(2.1538343756723917, 2.2653381826084606, -3.550087952059485, q[10])
        circuit.u3(1.307627685019188, -0.44686656993522567, -2.3238098554327418, q[6])
        circuit.u3(2.2046797998462906, 0.9732961754855436, 1.8527865921467421, q[9])
        circuit.u3(2.1665254613904126, -1.281337664694577, -1.2424905413631209, q[0])
        circuit.cx(q[0], q[9])
        circuit.u1(2.6209599970201007, q[9])
        circuit.u3(0.04680566321901303, 0.0, 0.0, q[0])
        circuit.cx(q[9], q[0])
        circuit.u3(1.7728411151289603, 0.0, 0.0, q[0])
        circuit.cx(q[0], q[9])
        circuit.u3(2.4866395967434443, 0.48684511243566697, -3.0069186877854728, q[9])
        circuit.u3(1.7369112924273789, -4.239660866163805, 1.0623389015296005, q[0])
        circuit.barrier(q)
        circuit.measure(q, c)

        try:
            qobj = qiskit._compiler.compile(circuit, backend)
        except QISKitError:
            qobj = None
        self.assertIsInstance(qobj, dict)

    @requires_qe_access
    def test_mapping_multi_qreg(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test mapping works for multiple qregs.
        """
        provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        backend = provider.get_backend('ibmqx5')

        q = qiskit.QuantumRegister(3, name='qr')
        q2 = qiskit.QuantumRegister(1, name='qr2')
        q3 = qiskit.QuantumRegister(4, name='qr3')
        c = qiskit.ClassicalRegister(3, name='cr')
        qc = qiskit.QuantumCircuit(q, q2, q3, c)
        qc.h(q[0])
        qc.cx(q[0], q2[0])
        qc.cx(q[1], q3[2])
        qc.measure(q, c)

        try:
            qobj = qiskit._compiler.compile(qc, backend)
        except QISKitError:
            qobj = None
        self.assertIsInstance(qobj, dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
