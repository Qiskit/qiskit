# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for expression and circuit variable substitution."""
from test import QiskitTestCase
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, WhileLoopOp
from qiskit.circuit.controlflow.switch_case import SwitchCaseOp, CASE_DEFAULT


class TestSubstituteVarInExpr(QiskitTestCase):
    """Tests for :func:`~.expr.substitute_var_in_expr`."""

    def test_substitute_var_in_binary(self):
        """A Var leaf is replaced throughout a binary expression."""
        var = expr.Var.new("a", types.Uint(8))
        node = expr.add(var, expr.lift(1, types.Uint(8)))
        replacement = expr.lift(5, types.Uint(8))
        result = expr.substitute_var_in_expr(node, {var: replacement})
        self.assertEqual(result, expr.add(replacement, expr.lift(1, types.Uint(8))))

    def test_substitute_var_in_range(self):
        """Substitution recurses into Range bounds."""
        var = expr.Var.new("stop", types.Uint(8))
        node = expr.Range(expr.lift(0, types.Uint(8)), var)
        replacement = expr.lift(10, types.Uint(8))
        result = expr.substitute_var_in_expr(node, {var: replacement})
        self.assertEqual(result, expr.Range(expr.lift(0, types.Uint(8)), replacement))

    def test_substitute_var_noop(self):
        """An empty substitution mapping returns the original node."""
        var = expr.Var.new("a", types.Uint(8))
        self.assertIs(expr.substitute_var_in_expr(var, {}), var)


class TestSubstituteVarInCircuit(QiskitTestCase):
    """Tests for :func:`~.expr.substitute_var_in_circuit`."""

    def test_substitute_var_in_store(self):
        """Store lvalues and rvalues are rewritten."""
        qc = QuantumCircuit()
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))
        qc.add_uninitialized_var(loop_var)
        qc.store(target, loop_var)

        result = expr.substitute_var_in_circuit(qc, {loop_var: expr.lift(3, types.Uint(8))})

        store_rvalues = [
            inst.operation.rvalue for inst in result.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(3, types.Uint(8)), store_rvalues)
        self.assertNotIn(loop_var, list(result.iter_vars()))

    def test_substitute_var_in_if_else_condition(self):
        """Classical if-else conditions are rewritten."""
        loop_var = expr.Var.new("i", types.Uint(8))
        true_body = QuantumCircuit(1)
        true_body.x(0)
        false_body = QuantumCircuit(1)
        qc = QuantumCircuit(1)
        qc.append(
            IfElseOp(
                expr.equal(loop_var, expr.lift(1, types.Uint(8))),
                true_body,
                false_body,
            ),
            [0],
        )

        result = expr.substitute_var_in_circuit(qc, {loop_var: expr.lift(1, types.Uint(8))})

        condition = result.data[0].operation.condition
        self.assertEqual(
            condition,
            expr.equal(expr.lift(1, types.Uint(8)), expr.lift(1, types.Uint(8))),
        )

    def test_substitute_var_in_for_loop_body(self):
        """For-loop bodies are rewritten by substituting loop variables."""
        loop_var = expr.Var.new("i", types.Uint(8))
        qc = QuantumCircuit(1)
        target = qc.add_var("target", expr.lift(0, types.Uint(8)))
        indexset = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))

        body = QuantumCircuit(1)
        body.add_uninitialized_var(loop_var)
        body.add_capture(target)
        body.store(target, loop_var)
        qc.append(ForLoopOp(indexset, loop_var, body), [0])

        result = expr.substitute_var_in_circuit(qc, {loop_var: expr.lift(2, types.Uint(8))})

        for_loop = next(inst.operation for inst in result.data if inst.operation.name == "for_loop")
        self.assertEqual(for_loop.params[0], indexset)
        self.assertEqual(for_loop.params[1], loop_var)
        body_stores = [
            inst.operation.rvalue
            for inst in for_loop.params[2].data
            if inst.operation.name == "store"
        ]
        self.assertEqual(body_stores, [expr.lift(2, types.Uint(8))])

    def test_empty_substitution_returns_independent_circuit(self):
        """An empty mapping returns a copy whose mutation does not affect the input."""
        qc = QuantumCircuit(1)
        qc.add_var("target", expr.lift(0, types.Uint(8)))
        qc.x(0)

        result = expr.substitute_var_in_circuit(qc, {})

        self.assertIsNot(result, qc)
        original_len = len(qc.data)
        result.x(0)
        self.assertEqual(len(qc.data), original_len)
        self.assertEqual(len(result.data), original_len + 1)

    def test_declared_var_initialization_preserved(self):
        """Initial values of preserved declared vars survive substitution."""
        qc = QuantumCircuit()
        qc.add_var("kept", expr.lift(7, types.Uint(8)))
        loop_var = expr.Var.new("i", types.Uint(8))
        qc.add_uninitialized_var(loop_var)

        result = expr.substitute_var_in_circuit(qc, {loop_var: expr.lift(3, types.Uint(8))})

        store_rvalues = [
            inst.operation.rvalue for inst in result.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(7, types.Uint(8)), store_rvalues)

    def test_substitute_var_in_switch_case_roundtrip(self):
        """Switch-case targets and bodies are rewritten and reconstructed correctly."""
        target_var = expr.Var.new("t", types.Uint(8))
        body_zero = QuantumCircuit(1)
        body_zero.x(0)
        body_default = QuantumCircuit(1)
        body_default.h(0)

        qc = QuantumCircuit(1)
        qc.append(
            SwitchCaseOp(target_var, [(0, body_zero), (CASE_DEFAULT, body_default)]),
            [0],
        )

        result = expr.substitute_var_in_circuit(qc, {target_var: expr.lift(0, types.Uint(8))})

        switch = result.data[0].operation
        self.assertEqual(switch.target, expr.lift(0, types.Uint(8)))
        cases = list(switch.cases_specifier())
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0][0], (0,))
        self.assertEqual(cases[1][0], (CASE_DEFAULT,))

    def test_for_loop_with_python_range_indexset_recurses_into_body(self):
        """For-loop whose indexset is a Python ``range`` (not an ``expr.Range``).

        The ``isinstance(indexset, expr.Range)`` guard in ``substitute.py`` means a Python
        ``range`` must pass through untouched.  This test additionally proves that taking
        that guarded branch does not skip recursion into the body — a captured Var used
        in a body ``Store`` is still substituted.

        Compile-time :class:`~.Parameter` loop variables are preserved by name only;
        object identity with the caller's original instance is not guaranteed (consistent
        with :meth:`~.QuantumCircuit.append` deepcopy behavior for parameterized ops).
        """
        captured = expr.Var.new("flag", types.Uint(8))
        loop_param = Parameter("i")

        body = QuantumCircuit(1)
        body.add_capture(captured)
        target = body.add_var("target", expr.lift(0, types.Uint(8)))
        body.store(target, captured)

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(captured)
        qc.append(ForLoopOp(range(4), loop_param, body), [0])

        result = expr.substitute_var_in_circuit(qc, {captured: expr.lift(7, types.Uint(8))})

        for_loop = next(inst.operation for inst in result.data if inst.operation.name == "for_loop")
        self.assertEqual(for_loop.params[0], range(4))
        self.assertIsInstance(for_loop.params[1], Parameter)
        self.assertEqual(for_loop.params[1].name, loop_param.name)
        body_stores = [
            inst.operation.rvalue
            for inst in for_loop.params[2].data
            if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(7, types.Uint(8)), body_stores)
        self.assertNotIn(captured, list(result.iter_vars()))

    def test_substitute_var_inside_if_nested_in_for_loop(self):
        """Vars in an if-condition nested inside a for-loop body are rewritten."""
        flag = expr.Var.new("flag", types.Uint(8))

        then_body = QuantumCircuit(1)
        then_body.x(0)

        for_body = QuantumCircuit(1)
        for_body.add_capture(flag)
        for_body.append(
            IfElseOp(expr.equal(flag, expr.lift(1, types.Uint(8))), then_body, None),
            [0],
        )

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(flag)
        qc.append(ForLoopOp(range(3), Parameter("i"), for_body), [0])

        result = expr.substitute_var_in_circuit(qc, {flag: expr.lift(1, types.Uint(8))})

        for_loop = result.data[-1].operation
        inner_if = for_loop.params[2].data[0].operation
        self.assertEqual(
            inner_if.condition,
            expr.equal(expr.lift(1, types.Uint(8)), expr.lift(1, types.Uint(8))),
        )
        self.assertNotIn(flag, list(result.iter_vars()))

    def test_substitute_var_inside_while_nested_in_if(self):
        """Vars in a while-condition nested inside an if-true block are rewritten."""
        cond_var = expr.Var.new("c", types.Uint(8))

        while_body = QuantumCircuit(1)
        while_body.x(0)

        then_body = QuantumCircuit(1)
        then_body.add_capture(cond_var)
        then_body.append(
            WhileLoopOp(expr.equal(cond_var, expr.lift(0, types.Uint(8))), while_body),
            [0],
        )

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(cond_var)
        qc.append(
            IfElseOp(expr.lift(True, types.Bool()), then_body, None),
            [0],
        )

        result = expr.substitute_var_in_circuit(qc, {cond_var: expr.lift(5, types.Uint(8))})

        if_op = result.data[-1].operation
        inner_while = if_op.blocks[0].data[0].operation
        self.assertEqual(
            inner_while.condition,
            expr.equal(expr.lift(5, types.Uint(8)), expr.lift(0, types.Uint(8))),
        )
        self.assertNotIn(cond_var, list(result.iter_vars()))

    def test_substitute_var_inside_switch_case_body(self):
        """Vars inside the body of a switch-case branch are rewritten."""
        target_var = expr.Var.new("t", types.Uint(8))
        rvalue_var = expr.Var.new("payload", types.Uint(8))

        case_zero = QuantumCircuit(1)
        case_zero.add_capture(rvalue_var)
        local = case_zero.add_var("local", expr.lift(0, types.Uint(8)))
        case_zero.store(local, rvalue_var)

        case_default = QuantumCircuit(1)
        case_default.h(0)

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(target_var)
        qc.add_uninitialized_var(rvalue_var)
        qc.append(
            SwitchCaseOp(target_var, [(0, case_zero), (CASE_DEFAULT, case_default)]),
            [0],
        )

        result = expr.substitute_var_in_circuit(
            qc,
            {
                target_var: expr.lift(0, types.Uint(8)),
                rvalue_var: expr.lift(9, types.Uint(8)),
            },
        )

        switch = result.data[-1].operation
        self.assertEqual(switch.target, expr.lift(0, types.Uint(8)))
        case_zero_body = next(body for labels, body in switch.cases_specifier() if 0 in labels)
        body_stores = [
            inst.operation.rvalue for inst in case_zero_body.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(9, types.Uint(8)), body_stores)
        self.assertNotIn(target_var, list(result.iter_vars()))
        self.assertNotIn(rvalue_var, list(result.iter_vars()))

    def test_substitute_var_through_three_level_nesting(self):
        """Three-level nesting: for > if > store — substitution reaches the deepest body."""
        deep_var = expr.Var.new("deep", types.Uint(8))

        innermost = QuantumCircuit(1)
        innermost.add_capture(deep_var)
        sink = innermost.add_var("sink", expr.lift(0, types.Uint(8)))
        innermost.store(sink, deep_var)

        if_block = QuantumCircuit(1)
        if_block.add_capture(deep_var)
        if_block.append(
            IfElseOp(expr.equal(deep_var, expr.lift(2, types.Uint(8))), innermost, None),
            [0],
        )

        qc = QuantumCircuit(1)
        qc.add_uninitialized_var(deep_var)
        qc.append(ForLoopOp(range(2), Parameter("i"), if_block), [0])

        result = expr.substitute_var_in_circuit(qc, {deep_var: expr.lift(2, types.Uint(8))})

        for_body = result.data[-1].operation.params[2]
        inner_if = for_body.data[0].operation
        self.assertEqual(
            inner_if.condition,
            expr.equal(expr.lift(2, types.Uint(8)), expr.lift(2, types.Uint(8))),
        )
        innermost_body = inner_if.blocks[0]
        innermost_stores = [
            inst.operation.rvalue for inst in innermost_body.data if inst.operation.name == "store"
        ]
        self.assertIn(expr.lift(2, types.Uint(8)), innermost_stores)
        self.assertNotIn(deep_var, list(result.iter_vars()))
