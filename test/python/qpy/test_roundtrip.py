# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for python write/rust read flow and vice versa"""

import io
import uuid

from ddt import ddt, idata, unpack

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Duration
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.random import random_circuit
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.classical import expr, types
from qiskit.synthesis import LieTrotter
from qiskit.qpy.common import QPY_RUST_READ_MIN_VERSION, QPY_RUST_WRITE_MIN_VERSION, QPY_VERSION
from qiskit.qpy.binary_io import write_circuit, read_circuit
from qiskit.qpy import UnsupportedFeatureForVersion
from test import QiskitTestCase


def all_qpy_combinations(min_version):
    def wrapper(func):
        return idata(
            (version, write_with, read_with)
            for version in range(min_version, QPY_VERSION + 1)
            for write_with in (
                ("Python", "Rust") if version >= QPY_RUST_WRITE_MIN_VERSION else ("Python",)
            )
            for read_with in (
                ("Python", "Rust") if version >= QPY_RUST_READ_MIN_VERSION else ("Python",)
            )
        )(unpack(func))

    return wrapper


@ddt
class TestQPYRoundtrip(QiskitTestCase):
    """QPY circuit testing platform."""

    def assert_roundtrip_equal(
        self,
        circuit,
        version,
        write_with,
        read_with,
        annotation_factories=None,
    ):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        use_rust_for_write = write_with == "Rust"
        use_rust_for_read = read_with == "Rust"
        write_circuit(
            qpy_file,
            circuit,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=use_rust_for_write,
        )
        qpy_file.seek(0)
        new_circuit = read_circuit(
            qpy_file,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=use_rust_for_read,
        )
        self.assertEqual(circuit, new_circuit)
        self.assertEqual(circuit.layout, new_circuit.layout)
        self.assertEqual(circuit.parameters, new_circuit.parameters)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_simple(self, version, write_with, read_with):
        """Basic roundtrip test"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_ifelse(self, version, write_with, read_with):
        """Check the IfElse control flow gate passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        condition = (qc.cregs[0], 0)
        body = QuantumCircuit([qc.qubits[0]])
        body.x(0)
        false_body = QuantumCircuit([qc.qubits[1]])
        false_body.y(0)
        qc.if_else(condition, body, false_body, [qc.qubits[0]], [])
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_box(self, version, write_with, read_with):
        """Check the BoxOp control flow gate passes roundtrip"""
        qc = QuantumCircuit(2)
        with qc.box(duration=13):
            qc.cx(0, 1)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

        # test with Expression duration
        qc = QuantumCircuit(2)
        a = qc.add_stretch("a")
        with qc.box(duration=expr.mul(2, a)):
            qc.cx(0, 1)
        if version < 14:
            with io.BytesIO() as fptr, self.assertRaises(UnsupportedFeatureForVersion):
                write_circuit(fptr, qc, version=version, use_rust=write_with == "Rust")
        else:
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_forloop(self, version, write_with, read_with):
        """Check the ForLoop control flow gate passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

        qc = QuantumCircuit(2, 1)
        with qc.for_loop((1, 4)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_nested_for_loop(self, version, write_with, read_with):
        """Check the nested ForLoop control flow gate passes roundtrip"""
        qc = QuantumCircuit(6, 6)
        with qc.if_test(expr.equal(qc.cregs[0], 1)) as else_:
            qc.cx(0, 1)
            qc.cz(0, 2)
            qc.cz(0, 3)
        with else_:
            qc.cz(1, 4)
            with qc.for_loop((1, 2)):
                qc.cx(1, 5)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_switch(self, version, write_with, read_with):
        """Check the SwitchOp control flow gate passes roundtrip"""
        body = QuantumCircuit(1)
        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy()), (2, body.copy())], [0], [])
        qc.switch(expr.logic_not(qc.clbits[0]), [(False, body.copy())], [0], [])
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_for_loop_with_range(self, version, write_with, read_with):
        """Check the ForLoop control flow gate with Range expression passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        start = expr.lift(0, types.Uint(8))
        stop = expr.lift(5, types.Uint(8))
        step = expr.lift(1, types.Uint(8))
        range_expr = expr.Range(start, stop, step)

        with qc.for_loop(range_expr):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_for_loop_with_var_loop_parameter(self, version, write_with, read_with):
        """For-loop expr.Var loop parameters round-trip faithfully in QPY v17+."""
        qc = QuantumCircuit(1)
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(6, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        with qc.for_loop(range_expr) as loop_var:
            qc.store(target, loop_var)

        if version >= 17:
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )
            qpy_file = io.BytesIO()
            write_circuit(
                qpy_file,
                qc,
                version=version,
                use_rust=write_with == "Rust",
            )
            qpy_file.seek(0)
            loaded = read_circuit(
                qpy_file,
                version=version,
                use_rust=read_with == "Rust",
            )
            from qiskit.qasm3 import dumps

            loop_var = next(
                inst.operation.params[1]
                for inst in loaded.data
                if inst.operation.name == "for_loop"
            )
            self.assertIn(f"for uint[8] {loop_var.name} in", dumps(loaded))
            return

        qpy_file = io.BytesIO()
        if write_with == "Python":
            write_circuit(
                qpy_file,
                qc,
                version=version,
                use_rust=False,
            )
        else:
            write_circuit(
                qpy_file,
                qc,
                version=version,
                use_rust=True,
            )
        qpy_file.seek(0)
        loaded = read_circuit(
            qpy_file,
            version=version,
            use_rust=read_with == "Rust",
        )

        original_for_loop = next(
            inst.operation for inst in qc.data if inst.operation.name == "for_loop"
        )
        loaded_for_loop = next(
            inst.operation for inst in loaded.data if inst.operation.name == "for_loop"
        )
        self.assertEqual(original_for_loop.params[0], loaded_for_loop.params[0])
        self.assertIsInstance(original_for_loop.params[1], expr.Var)
        self.assertIsNone(loaded_for_loop.params[1])

    @all_qpy_combinations(17)
    def test_for_loop_with_explicit_unused_var_loop_parameter(self, version, write_with, read_with):
        """Explicit loop Vars are preserved even when unused in the body."""
        loop_var = expr.Var.new("i", types.Uint(8))
        qc = QuantumCircuit(1)
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)), expr.lift(1, types.Uint(8))
        )
        with qc.for_loop(range_expr, loop_var):
            qc.h(0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(17)
    def test_for_loop_with_dynamic_range_and_var_loop_parameter(
        self, version, write_with, read_with
    ):
        """Dynamic expr.Range for-loops with loop Vars round-trip in QPY v17+."""
        qc = QuantumCircuit(1)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(6, types.Uint(8)))
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        range_expr = expr.Range(start_var, stop_var)
        with qc.for_loop(range_expr) as loop_var:
            qc.store(target, expr.Cast(loop_var, types.Uint(8)))
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    def test_for_loop_with_var_loop_parameter_v16_regression(self):
        """QPY v16 and below still drop runtime loop variables from the param slot."""
        qc = QuantumCircuit(1)
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(6, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        with qc.for_loop(range_expr) as loop_var:
            qc.store(target, loop_var)

        qpy_file = io.BytesIO()
        write_circuit(qpy_file, qc, version=16, use_rust=False)
        qpy_file.seek(0)
        loaded = read_circuit(qpy_file, version=16, use_rust=False)

        loaded_for_loop = next(
            inst.operation for inst in loaded.data if inst.operation.name == "for_loop"
        )
        self.assertIsNone(loaded_for_loop.params[1])
        self.assertIsInstance(loop_var, expr.Var)

    def test_for_loop_v16_body_remains_coherent_after_loop_var_dropped(self):
        """At QPY v16 the loop Var is dropped from param[1] but the body is still valid.

        The body has its own standalone-var section that includes the loop Var as a
        declared variable, so Store instructions that reference it deserialise correctly
        and the Var can be resolved inside the body scope.
        """
        qc = QuantumCircuit(1)
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))
        with qc.for_loop(range_expr) as loop_var:
            qc.store(target, loop_var)

        qpy_file = io.BytesIO()
        write_circuit(qpy_file, qc, version=16, use_rust=False)
        qpy_file.seek(0)
        loaded = read_circuit(qpy_file, version=16, use_rust=False)

        loaded_for_loop = next(
            inst.operation for inst in loaded.data if inst.operation.name == "for_loop"
        )
        self.assertIsNone(loaded_for_loop.params[1])
        body = loaded_for_loop.params[2]
        body_stores = [inst.operation for inst in body.data if inst.operation.name == "store"]
        self.assertEqual(len(body_stores), 1)
        self.assertIsInstance(body_stores[0].rvalue, expr.Var)
        # The Var referenced in the Store must be declared somewhere in the body scope.
        self.assertIn(body_stores[0].rvalue, set(body.iter_vars()))

    def test_for_loop_dynamic_range_v16_preserves_indexset_drops_loop_var(self):
        """Dynamic expr.Range (Var-bounded) indexset survives a v16 round-trip.

        expr.Range is serialised as a Value.EXPRESSION (existing type key) at all
        versions, so the Var-bounded bounds are preserved even at v16.  Only the loop
        Var in param[1] is dropped (NULL), which is the documented lossy behaviour.
        """
        qc = QuantumCircuit(1)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(6, types.Uint(8)))
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        range_expr = expr.Range(start_var, stop_var)
        with qc.for_loop(range_expr) as loop_var:
            qc.store(target, expr.Cast(loop_var, types.Uint(8)))

        qpy_file = io.BytesIO()
        write_circuit(qpy_file, qc, version=16, use_rust=False)
        qpy_file.seek(0)
        loaded = read_circuit(qpy_file, version=16, use_rust=False)

        loaded_for_loop = next(
            inst.operation for inst in loaded.data if inst.operation.name == "for_loop"
        )
        # Var-bounded indexset is preserved end-to-end.
        self.assertEqual(loaded_for_loop.params[0], range_expr)
        # Loop Var dropped at v16 — documented lossy behaviour.
        self.assertIsNone(loaded_for_loop.params[1])

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_for_loop_with_expr_range_and_none_loop_parameter(self, version, write_with, read_with):
        """ForLoopOp with expr.Range indexset and explicit None loop parameter round-trips.

        When the user opts out of a loop variable (param[1] is None), the ForLoopOp
        still serialises and deserialises correctly across all supported QPY versions.
        This exercises the NULL param path for ForLoopOp param[1] with an expr.Range
        indexset, distinct from the LOOP_VARIABLE path and the Python-range NULL path.
        """
        qc = QuantumCircuit(1)
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))
        # Explicit None: no loop variable requested.
        from qiskit.circuit.controlflow import ForLoopOp

        body = QuantumCircuit(1)
        body.h(0)
        qc.append(ForLoopOp(range_expr, None, body), [0])
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_evolutiongate(self, version, write_with, read_with):
        """Test loading a circuit with evolution gate works."""
        synthesis = LieTrotter(reps=2)
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=2, synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_parameter_expression(self, version, write_with, read_with):
        """Test loading a circuit with parameter expression works"""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_param = theta + phi
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(sum_param, range(3))
        qc.rz(phi, 3)
        qc.rz(theta, 4)
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_parameter_expression_with_vectors(self, version, write_with, read_with):
        """Test loading a circuit with parameter expression works"""
        theta = ParameterVector("θ", 3)
        beta = Parameter("β")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(beta + theta[1], qr)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_parameter_expression_subs(self, version, write_with, read_with):
        """Test loading a circuit with parameter substitution works"""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        b = Parameter("b")
        exp = a + b
        exp = exp.subs({b: a})
        qc.ry(exp, 0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_degenerate_parameter_expression(self, version, write_with, read_with):
        """Test a circuit with a parameter expression that simplifies to 0."""
        x = Parameter("x")
        y_vec = ParameterVector("y", 2)
        z = Parameter("z")
        cases = [0 * x, 0 * x + 2, 0 * x + z, x - x, 0 * y_vec[0], 0 * (x + y_vec[1])]
        for case in cases:
            qc = QuantumCircuit(1)
            qc.rz(case, 0)
            self.assert_roundtrip_equal(
                qc, version=version, write_with=write_with, read_with=read_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_random_circuits(self, version, write_with, read_with):
        """Test loading a random circuit works"""
        for i in range(10):
            qc = random_circuit(10, 10, measure=True, conditional=True, reset=True, seed=42 + i)
            # Make sure the circuits round-trip as a sanity check
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_delay_roundtrip(self, version, write_with, read_with):
        qc = QuantumCircuit(1)
        qc.delay(1, 0, "dt")
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(14)
    def test_delay_expr_roundtrip(self, version, write_with, read_with):
        stretch_expr = QuantumCircuit(1, name="stretch_expr_delay_circuit")
        s = expr.Stretch(uuid.uuid4(), "a")
        stretch = stretch_expr.add_stretch(s)
        stretch_expr.delay(stretch, 0)
        stretch_expr.delay(expr.add(Duration.dt(200), stretch), 0)
        stretch_expr.delay(expr.sub(Duration.ns(3.14159), stretch), 0)
        self.assert_roundtrip_equal(
            stretch_expr, version=version, read_with=read_with, write_with=write_with
        )

    @all_qpy_combinations(14)
    def test_box_expr_roundtrip(self, version, write_with, read_with):
        qc = QuantumCircuit(1, name="box_expr_circuit")
        s = qc.add_stretch("s")
        duration = expr.add(Duration.dt(100), expr.sub(s, Duration.ns(16.25)))
        with qc.box(duration=duration):
            qc.x(0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_literal_integers_in_for(self, version, write_with, read_with):
        qc = QuantumCircuit(1)
        with qc.for_loop((2, 5, (1 << 60))) as _:
            qc.x(0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)
