# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

import math

from test import combine

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile
from qiskit.compiler.transpiler import _parse_inst_map
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.test.base import QiskitTestCase
from qiskit.test.mock.fake_backend_v2 import (
    FakeBackendV2,
    FakeBackend5QV2,
    FakeBackendSimple,
    FakeBackendV2LegacyQubitProps,
)
from qiskit.test.mock.fake_mumbai_v2 import FakeMumbaiFractionalCX
from qiskit.quantum_info import Operator


@ddt
class TestBackendV2(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.backend = FakeBackendV2()

    def assertMatchesTargetConstraints(self, tqc, target):
        qubit_indices = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        for instr, qargs, _ in tqc.data:
            qargs = tuple(qubit_indices[x] for x in qargs)
            target_set = target[instr.name].keys()
            self.assertIn(
                qargs,
                target_set,
                f"qargs: {qargs} not found in target for operation {instr.name}: {set(target_set)}",
            )

    def test_qubit_properties(self):
        """Test that qubit properties are returned as expected."""
        props = self.backend.qubit_properties([1, 0])
        self.assertEqual([73.09352e-6, 63.48783e-6], [x.t1 for x in props])
        self.assertEqual([126.83382e-6, 112.23246e-6], [x.t2 for x in props])
        self.assertEqual([5.26722e9, 5.17538e9], [x.frequency for x in props])

    def test_legacy_qubit_properties(self):
        """Test that qubit props work for backends not using properties in target."""
        props = FakeBackendV2LegacyQubitProps().qubit_properties([1, 0])
        self.assertEqual([73.09352e-6, 63.48783e-6], [x.t1 for x in props])
        self.assertEqual([126.83382e-6, 112.23246e-6], [x.t2 for x in props])
        self.assertEqual([5.26722e9, 5.17538e9], [x.frequency for x in props])

    def test_no_qubit_properties_raises(self):
        """Ensure that if no qubit properties are defined we raise correctly."""
        with self.assertRaises(NotImplementedError):
            FakeBackendSimple().qubit_properties(0)

    def test_option_bounds(self):
        """Test that option bounds are enforced."""
        with self.assertRaises(ValueError) as cm:
            self.backend.set_options(shots=8192)
        self.assertEqual(
            str(cm.exception),
            "Specified value for 'shots' is not a valid value, must be >=1 or <=4096",
        )

    @data(0, 1, 2, 3)
    def test_transpile(self, opt_level):
        """Test that transpile() works with a BackendV2 backend."""
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.cz(1, 0)
        with self.assertLogs("qiskit.providers.backend", level="WARN") as log:
            tqc = transpile(qc, self.backend, optimization_level=opt_level)
        self.assertEqual(
            log.output,
            [
                "WARNING:qiskit.providers.backend:This backend's operations: "
                "cx,ecr only apply to a subset of qubits. Using this property to "
                "get 'basis_gates' for the transpiler may potentially create "
                "invalid output"
            ],
        )
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        self.assertMatchesTargetConstraints(tqc, self.backend.target)

    @combine(
        opt_level=[0, 1, 2, 3],
        gate=["cx", "ecr", "cz"],
        bidirectional=[True, False],
        dsc=(
            "Test GHZ circuit with {gate} using opt level {opt_level} on backend "
            "with bidirectional={bidirectional}"
        ),
        name="{gate}_level_{opt_level}_bidirectional_{bidirectional}",
    )
    def test_5q_ghz(self, opt_level, gate, bidirectional):
        backend = FakeBackend5QV2(bidirectional)
        qc = QuantumCircuit(5)
        qc.h(0)
        getattr(qc, gate)(0, 1)
        getattr(qc, gate)(2, 1)
        getattr(qc, gate)(2, 3)
        getattr(qc, gate)(4, 3)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        t_op = Operator.from_circuit(tqc)
        self.assertTrue(t_op.equiv(qc))
        self.assertMatchesTargetConstraints(tqc, backend.target)

    def test_transpile_respects_arg_constraints(self):
        """Test that transpile() respects a heterogenous basis."""
        # Test CX on wrong link
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(1, 0)
        tqc = transpile(qc, self.backend)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        # Below is done to check we're decomposing cx(1, 0) with extra
        # rotations to correct for direction. However because of fp
        # differences between windows and other platforms the optimization
        # from the 1q optimization passes differ and the output gates
        # change (while still being equivalent). This relaxes the check to
        # still ensure it's valid but not so specific that it fails on windows
        self.assertEqual(tqc.count_ops().keys(), {"cx", "u"})
        self.assertEqual(tqc.count_ops()["cx"], 1)
        self.assertLessEqual(tqc.count_ops()["u"], 4)
        self.assertMatchesTargetConstraints(tqc, self.backend.target)
        # Test ECR on wrong link
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ecr(0, 1)
        tqc = transpile(qc, self.backend)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        self.assertEqual(tqc.count_ops(), {"ecr": 1, "u": 4})
        self.assertMatchesTargetConstraints(tqc, self.backend.target)

    def test_transpile_relies_on_gate_direction(self):
        """Test that transpile() relies on gate direction pass for 2q."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ecr(0, 1)
        tqc = transpile(qc, self.backend)
        expected = QuantumCircuit(2)
        expected.u(0, 0, -math.pi, 0)
        expected.u(math.pi / 2, 0, 0, 1)
        expected.ecr(1, 0)
        expected.u(math.pi / 2, 0, -math.pi, 0)
        expected.u(math.pi / 2, 0, -math.pi, 1)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        self.assertEqual(tqc.count_ops(), {"ecr": 1, "u": 4})
        self.assertMatchesTargetConstraints(tqc, self.backend.target)

    def test_transpile_mumbai_target(self):
        """Test that transpile respects a more involved target for a fake mumbai."""
        backend = FakeMumbaiFractionalCX()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(1, 0)
        qc.measure_all()
        tqc = transpile(qc, backend)
        qr = QuantumRegister(27, "q")
        cr = ClassicalRegister(2, "meas")
        expected = QuantumCircuit(qr, cr, global_phase=math.pi / 4)
        expected.rz(math.pi / 2, 0)
        expected.sx(0)
        expected.rz(math.pi / 2, 0)
        expected.cx(1, 0)
        expected.barrier(qr[0:2])
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[1])
        self.assertEqual(expected, tqc)

    def test_transpile_parse_inst_map(self):
        """Test that transpiler._parse_inst_map() supports BackendV2."""
        inst_map = _parse_inst_map(inst_map=None, backend=self.backend, num_circuits=1)[0]
        self.assertIsInstance(inst_map, InstructionScheduleMap)
