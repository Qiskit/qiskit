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

"""Expression-tree variable substitution utilities.

This module is the supported entry point for rewriting classical :class:`~.expr.Var` nodes
inside an expression or a :class:`~.QuantumCircuit`.  Transpiler passes that need this
functionality should wrap these utilities rather than reimplementing substitution on a
:class:`~.DAGCircuit`: control-flow bodies are stored as nested circuits, so substitution
is fundamentally a circuit-level walk and a DAG round-trip adds no value.
"""

from __future__ import annotations

__all__ = ["substitute_var_in_circuit", "substitute_var_in_expr"]

from typing import TYPE_CHECKING

from . import expr as _expr

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def substitute_var_in_expr(
    node: _expr.Expr, substitutions: dict[_expr.Var, _expr.Expr]
) -> _expr.Expr:
    """Return a copy of *node* with every :class:`~.expr.Var` in *substitutions* replaced.

    Nodes not mentioned in *substitutions* are returned unchanged.  The function is purely
    structural: types are preserved from the original tree, not re-inferred.

    Args:
        node: the expression to rewrite.
        substitutions: mapping from :class:`~.expr.Var` to the replacement
            :class:`~.expr.Expr`.

    Returns:
        A new expression tree with the substitutions applied, or *node* itself if
        nothing changed.
    """
    if not substitutions:
        return node
    return _substitute(node, substitutions)


def _substitute(node: _expr.Expr, subs: dict) -> _expr.Expr:
    if isinstance(node, _expr.Var):
        return subs.get(node, node)
    if isinstance(node, (_expr.Value, _expr.Stretch)):
        return node
    if isinstance(node, _expr.Cast):
        new_operand = _substitute(node.operand, subs)
        if new_operand is node.operand:
            return node
        return _expr.Cast(new_operand, node.type, implicit=node.implicit)
    if isinstance(node, _expr.Unary):
        new_operand = _substitute(node.operand, subs)
        if new_operand is node.operand:
            return node
        return _expr.Unary(node.op, new_operand, node.type)
    if isinstance(node, _expr.Binary):
        new_left = _substitute(node.left, subs)
        new_right = _substitute(node.right, subs)
        if new_left is node.left and new_right is node.right:
            return node
        return _expr.Binary(node.op, new_left, new_right, node.type)
    if isinstance(node, _expr.Index):
        new_target = _substitute(node.target, subs)
        new_index = _substitute(node.index, subs)
        if new_target is node.target and new_index is node.index:
            return node
        return _expr.Index(new_target, new_index, node.type)
    if isinstance(node, _expr.Range):
        new_start = _substitute(node.start, subs)
        new_stop = _substitute(node.stop, subs)
        new_step = _substitute(node.step, subs)
        if new_start is node.start and new_stop is node.stop and new_step is node.step:
            return node
        return _expr.Range(new_start, new_stop, new_step, node.type)
    return node


def substitute_var_in_circuit(
    circuit: QuantumCircuit, substitutions: dict[_expr.Var, _expr.Expr]
) -> QuantumCircuit:
    """Return a copy of *circuit* with every :class:`~.expr.Var` in *substitutions* replaced.

    The substituted variables are removed from the copy's declared variables.  Any classical
    expression in the circuit — :class:`~.Store` lvalues and rvalues, control-flow conditions,
    and switch targets — has the substitution applied recursively.  Nested control-flow bodies
    are also rewritten.

    The returned circuit is independent of *circuit*: instructions are replayed into a fresh
    shell created by :meth:`~.QuantumCircuit.copy_empty_like`, so mutating the result will
    not affect the input.  Declared variables that are preserved keep their initial values
    because the original initializing :class:`~.Store` operations are part of ``circuit.data``
    and are replayed by the instruction loop below.

    Args:
        circuit: the circuit to rewrite.
        substitutions: mapping from :class:`~.expr.Var` to the replacement
            :class:`~.expr.Expr`.

    Returns:
        A new :class:`~.QuantumCircuit` with the substitutions applied.
    """
    return _substitute_circuit(circuit, substitutions)


def _substitute_circuit(circuit: QuantumCircuit, subs: dict) -> QuantumCircuit:
    from qiskit.circuit.store import Store
    from qiskit.circuit.controlflow import (
        IfElseOp,
        WhileLoopOp,
        ForLoopOp,
    )
    from qiskit.circuit.controlflow.switch_case import SwitchCaseOp

    # ``vars_mode="drop"`` gives us an empty shell with the same qubits/clbits/registers
    # and global state, but no realtime variables — we then re-add only the ones that
    # aren't being substituted away.
    out = circuit.copy_empty_like(vars_mode="drop")

    for var in circuit.iter_input_vars():
        if var not in subs:
            out.add_input(var)
    for var in circuit.iter_captured_vars():
        if var not in subs:
            out.add_capture(var)
    for var in circuit.iter_declared_vars():
        if var not in subs:
            out.add_uninitialized_var(var)

    for inst in circuit.data:
        op = inst.operation
        if isinstance(op, Store):
            new_op = Store(
                _substitute(op.lvalue, subs),
                _substitute(op.rvalue, subs),
            )
        elif isinstance(op, IfElseOp):
            new_cond = (
                _substitute(op.condition, subs)
                if isinstance(op.condition, _expr.Expr)
                else op.condition
            )
            new_blocks = [_substitute_circuit(b, subs) for b in op.blocks]
            new_op = IfElseOp(
                new_cond,
                new_blocks[0],
                new_blocks[1] if len(new_blocks) > 1 else None,
                label=op.label,
            )
        elif isinstance(op, WhileLoopOp):
            new_cond = (
                _substitute(op.condition, subs)
                if isinstance(op.condition, _expr.Expr)
                else op.condition
            )
            new_body = _substitute_circuit(op.blocks[0], subs)
            new_op = WhileLoopOp(new_cond, new_body, label=op.label)
        elif isinstance(op, SwitchCaseOp):
            new_target = (
                _substitute(op.target, subs) if isinstance(op.target, _expr.Expr) else op.target
            )
            new_cases = [
                (case, _substitute_circuit(body, subs)) for case, body in op.cases_specifier()
            ]
            new_op = SwitchCaseOp(new_target, new_cases, label=op.label)
        elif isinstance(op, ForLoopOp):
            indexset, loop_param, body = op.params
            new_indexset = (
                _substitute(indexset, subs) if isinstance(indexset, _expr.Range) else indexset
            )
            new_body = _substitute_circuit(body, subs)
            if new_indexset is indexset and new_body is body:
                new_op = op
            elif new_indexset is indexset:
                # Reuse ``params[1]`` from the original op so compile-time Parameters keep
                # object identity (see ``Parameter.__deepcopy__`` invariants).
                new_op = op.replace_blocks([new_body])
            else:
                new_op = ForLoopOp(new_indexset, loop_param, new_body, label=op.label)
        else:
            new_op = op

        out.append(new_op, inst.qubits, inst.clbits, copy=False)

    out.global_phase = circuit.global_phase
    return out
