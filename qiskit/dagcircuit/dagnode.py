# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Objects to represent the information at a node in the DAGCircuit."""

import itertools
import uuid
from typing import Iterable

from qiskit.circuit import (
    Qubit,
    Clbit,
    ClassicalRegister,
    ControlFlowOp,
    IfElseOp,
    WhileLoopOp,
    SwitchCaseOp,
    ForLoopOp,
    Parameter,
)
from qiskit.circuit.classical import expr


def _legacy_condition_eq(cond1, cond2, bit_indices1, bit_indices2):
    if cond1 is cond2 is None:
        return True
    elif None in (cond1, cond2):
        return False
    target1, val1 = cond1
    target2, val2 = cond2
    if val1 != val2:
        return False
    if isinstance(target1, Clbit) and isinstance(target2, Clbit):
        return bit_indices1[target1] == bit_indices2[target2]
    if isinstance(target1, ClassicalRegister) and isinstance(target2, ClassicalRegister):
        return target1.size == target2.size and all(
            bit_indices1[t1] == bit_indices2[t2] for t1, t2 in zip(target1, target2)
        )
    return False


def _circuit_to_dag(circuit, node_qargs, node_cargs, bit_indices):
    """Get a :class:`.DAGCircuit` of the given :class:`.QuantumCircuit`.  The bits in the output
    will be ordered in a canonical order based on their indices in the outer DAG, as defined by the
    ``bit_indices`` mapping and the ``node_{q,c}args`` arguments."""
    from qiskit.converters import circuit_to_dag  # pylint: disable=cyclic-import

    def sort_key(bits):
        outer, _inner = bits
        return bit_indices[outer]

    return circuit_to_dag(
        circuit,
        copy_operations=False,
        qubit_order=[
            inner for _outer, inner in sorted(zip(node_qargs, circuit.qubits), key=sort_key)
        ],
        clbit_order=[
            inner for _outer, inner in sorted(zip(node_cargs, circuit.clbits), key=sort_key)
        ],
    )


def _make_expr_key(bit_indices):
    def key(var):
        if isinstance(var, Clbit):
            return bit_indices.get(var)
        if isinstance(var, ClassicalRegister):
            return [bit_indices.get(bit) for bit in var]
        return None

    return key


def _condition_op_eq(node1, node2, bit_indices1, bit_indices2):
    cond1 = node1.op.condition
    cond2 = node2.op.condition
    if isinstance(cond1, expr.Expr) and isinstance(cond2, expr.Expr):
        if not expr.structurally_equivalent(
            cond1, cond2, _make_expr_key(bit_indices1), _make_expr_key(bit_indices2)
        ):
            return False
    elif isinstance(cond1, expr.Expr) or isinstance(cond2, expr.Expr):
        return False
    elif not _legacy_condition_eq(cond1, cond2, bit_indices1, bit_indices2):
        return False
    return len(node1.op.blocks) == len(node2.op.blocks) and all(
        _circuit_to_dag(block1, node1.qargs, node1.cargs, bit_indices1)
        == _circuit_to_dag(block2, node2.qargs, node2.cargs, bit_indices2)
        for block1, block2 in zip(node1.op.blocks, node2.op.blocks)
    )


def _switch_case_eq(node1, node2, bit_indices1, bit_indices2):
    target1 = node1.op.target
    target2 = node2.op.target
    if isinstance(target1, expr.Expr) and isinstance(target2, expr.Expr):
        if not expr.structurally_equivalent(
            target1, target2, _make_expr_key(bit_indices1), _make_expr_key(bit_indices2)
        ):
            return False
    elif isinstance(target1, Clbit) and isinstance(target2, Clbit):
        if bit_indices1[target1] != bit_indices2[target2]:
            return False
    elif isinstance(target1, ClassicalRegister) and isinstance(target2, ClassicalRegister):
        if target1.size != target2.size or any(
            bit_indices1[b1] != bit_indices2[b2] for b1, b2 in zip(target1, target2)
        ):
            return False
    else:
        return False
    cases1 = [case for case, _ in node1.op.cases_specifier()]
    cases2 = [case for case, _ in node2.op.cases_specifier()]
    return (
        len(cases1) == len(cases2)
        and all(set(labels1) == set(labels2) for labels1, labels2 in zip(cases1, cases2))
        and len(node1.op.blocks) == len(node2.op.blocks)
        and all(
            _circuit_to_dag(block1, node1.qargs, node1.cargs, bit_indices1)
            == _circuit_to_dag(block2, node2.qargs, node2.cargs, bit_indices2)
            for block1, block2 in zip(node1.op.blocks, node2.op.blocks)
        )
    )


def _for_loop_eq(node1, node2, bit_indices1, bit_indices2):
    indexset1, param1, body1 = node1.op.params
    indexset2, param2, body2 = node2.op.params
    if indexset1 != indexset2:
        return False
    if (param1 is None and param2 is not None) or (param1 is not None and param2 is None):
        return False
    if param1 is not None and param2 is not None:
        sentinel = Parameter(str(uuid.uuid4()))
        body1 = (
            body1.assign_parameters({param1: sentinel}, inplace=False)
            if param1 in body1.parameters
            else body1
        )
        body2 = (
            body2.assign_parameters({param2: sentinel}, inplace=False)
            if param2 in body2.parameters
            else body2
        )
    return _circuit_to_dag(body1, node1.qargs, node1.cargs, bit_indices1) == _circuit_to_dag(
        body2, node2.qargs, node2.cargs, bit_indices2
    )


_SEMANTIC_EQ_CONTROL_FLOW = {
    IfElseOp: _condition_op_eq,
    WhileLoopOp: _condition_op_eq,
    SwitchCaseOp: _switch_case_eq,
    ForLoopOp: _for_loop_eq,
}

_SEMANTIC_EQ_SYMMETRIC = frozenset({"barrier", "swap", "break_loop", "continue_loop"})


class DAGNode:
    """Parent class for DAGOpNode, DAGInNode, and DAGOutNode."""

    __slots__ = ["_node_id"]

    def __init__(self, nid=-1):
        """Create a node"""
        self._node_id = nid

    def __lt__(self, other):
        return self._node_id < other._node_id

    def __gt__(self, other):
        return self._node_id > other._node_id

    def __str__(self):
        # TODO is this used anywhere other than in DAG drawing?
        # needs to be unique as it is what pydot uses to distinguish nodes
        return str(id(self))

    @staticmethod
    def semantic_eq(node1, node2, bit_indices1, bit_indices2):
        """
        Check if DAG nodes are considered equivalent, e.g., as a node_match for
        :func:`rustworkx.is_isomorphic_node_match`.

        Args:
            node1 (DAGOpNode, DAGInNode, DAGOutNode): A node to compare.
            node2 (DAGOpNode, DAGInNode, DAGOutNode): The other node to compare.
            bit_indices1 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node1
            bit_indices2 (dict): Dictionary mapping Bit instances to their index
                within the circuit containing node2

        Return:
            Bool: If node1 == node2
        """
        if not isinstance(node1, DAGOpNode) or not isinstance(node1, DAGOpNode):
            return type(node1) is type(node2) and bit_indices1.get(node1.wire) == bit_indices2.get(
                node2.wire
            )
        if isinstance(node1.op, ControlFlowOp) and isinstance(node2.op, ControlFlowOp):
            # While control-flow operations aren't represented natively in the DAG, we have to do
            # some unpleasant dispatching and very manual handling.  Once they have more first-class
            # support we'll still be dispatching, but it'll look more appropriate (like the dispatch
            # based on `DAGOpNode`/`DAGInNode`/`DAGOutNode` that already exists) and less like we're
            # duplicating code from the `ControlFlowOp` classes.
            if type(node1.op) is not type(node2.op):
                return False
            comparer = _SEMANTIC_EQ_CONTROL_FLOW.get(type(node1.op))
            if comparer is None:  # pragma: no cover
                raise RuntimeError(f"unhandled control-flow operation: {type(node1.op)}")
            return comparer(node1, node2, bit_indices1, bit_indices2)

        node1_qargs = [bit_indices1[qarg] for qarg in node1.qargs]
        node1_cargs = [bit_indices1[carg] for carg in node1.cargs]

        node2_qargs = [bit_indices2[qarg] for qarg in node2.qargs]
        node2_cargs = [bit_indices2[carg] for carg in node2.cargs]

        # For barriers, qarg order is not significant so compare as sets
        if node1.op.name == node2.op.name and node1.name in _SEMANTIC_EQ_SYMMETRIC:
            node1_qargs = set(node1_qargs)
            node1_cargs = set(node1_cargs)
            node2_qargs = set(node2_qargs)
            node2_cargs = set(node2_cargs)

        return (
            node1_qargs == node2_qargs
            and node1_cargs == node2_cargs
            and _legacy_condition_eq(
                getattr(node1.op, "condition", None),
                getattr(node2.op, "condition", None),
                bit_indices1,
                bit_indices2,
            )
            and node1.op == node2.op
        )


class DAGOpNode(DAGNode):
    """Object to represent an Instruction at a node in the DAGCircuit."""

    __slots__ = ["op", "qargs", "cargs", "sort_key"]

    def __init__(self, op, qargs: Iterable[Qubit] = (), cargs: Iterable[Clbit] = (), dag=None):
        """Create an Instruction node"""
        super().__init__()
        self.op = op
        self.qargs = tuple(qargs)
        self.cargs = tuple(cargs)
        if dag is not None:
            cache_key = (self.qargs, self.cargs)
            key = dag._key_cache.get(cache_key, None)
            if key is not None:
                self.sort_key = key
            else:
                self.sort_key = ",".join(
                    f"{dag.find_bit(q).index:04d}" for q in itertools.chain(*cache_key)
                )
                dag._key_cache[cache_key] = self.sort_key
        else:
            self.sort_key = str(self.qargs)

    @property
    def name(self):
        """Returns the Instruction name corresponding to the op for this node"""
        return self.op.name

    @name.setter
    def name(self, new_name):
        """Sets the Instruction name corresponding to the op for this node"""
        self.op.name = new_name

    def __repr__(self):
        """Returns a representation of the DAGOpNode"""
        return f"DAGOpNode(op={self.op}, qargs={self.qargs}, cargs={self.cargs})"


class DAGInNode(DAGNode):
    """Object to represent an incoming wire node in the DAGCircuit."""

    __slots__ = ["wire", "sort_key"]

    def __init__(self, wire):
        """Create an incoming node"""
        super().__init__()
        self.wire = wire
        # TODO sort_key which is used in dagcircuit.topological_nodes
        # only works as str([]) for DAGInNodes. Need to figure out why.
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGInNode"""
        return f"DAGInNode(wire={self.wire})"


class DAGOutNode(DAGNode):
    """Object to represent an outgoing wire node in the DAGCircuit."""

    __slots__ = ["wire", "sort_key"]

    def __init__(self, wire):
        """Create an outgoing node"""
        super().__init__()
        self.wire = wire
        # TODO sort_key which is used in dagcircuit.topological_nodes
        # only works as str([]) for DAGOutNodes. Need to figure out why.
        self.sort_key = str([])

    def __repr__(self):
        """Returns a representation of the DAGOutNode"""
        return f"DAGOutNode(wire={self.wire})"
