# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Collapse chains of single qubit gates into a 2x2 matrix operator.
"""

from itertools import groupby
from functools import reduce

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator
from qiskit.circuit.exceptions import CircuitError


class Collapse1qChains(TransformationPass):
    """Collapse every chain of single-qubit gates into a 2x2 matrix operator.

    The resulting operators can be synthesized and optimized over the desired
    basis by invoking some follow-on passes.

    If the chain evaluates to identity (e.g. U(0,0,0)), this pass simply
    collapses the chain to none.

    The pass optionally accepts a basis_gates arg, whose purpose is to signal
    which gates can be executed by the backend. This information, paired with
    what basis the one-qubit synthesizer is able to synthesize into, will be
    used to decide which gates to collapse. If a gate in the circuit is already
    valid for the backend basis_gates, and the one-qubit synthesizer is not able
    to synthesize over it later, then that gate will be kept as-is (because it
    may be impossible to resynthesize that gate once it is collapsed). This may
    come at the cost of some optimization. For example, the synthesizer is able
    to target the Phase + Square-root(X) basis, but not any basis containing H.
    Thus given the following circuit:

    .. parsed-literal::
             ┌───┐┌───┐┌───┐
        q_0: ┤ H ├┤ H ├┤ H ├
             └───┘└───┘└───┘

    if basis_gates=['h'], this pass will not alter the circuit. On the other
    hand, if basis_gates=['p', 'sx'], this pass will collapse the chain into
    a unitary which will later synthesize into:

    .. parsed-literal::

             ┌─────────┐┌────┐┌─────────┐
        q_0: ┤ P(pi/2) ├┤ √X ├┤ P(pi/2) ├
             └─────────┘└────┘└─────────┘
    """
    def __init__(self, ignore_solo=False, basis_gates=None):
        """
        Args:
            ignore_solo (bool): If True, all solo gates (chains of length one)
                are left untouched (faster). Otherwise, each will be converted
                to a 2x2 matrix operator.
            basis_gates (list[str]): the set of gates that are valid to keep
                (i.e. target backend can execute them). If None, then all gates
                will be considered for collapse.
        """
        self.ignore_solo = ignore_solo
        self.basis_gates = basis_gates
        super().__init__()

    def run(self, dag):
        """Run the Collapse1qChains pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: a DAG with no single-qubit gate chains and only as single-qubit gates.
        """
        chains = []

        # collect chains of uninterrupted single-qubit gates
        topo_ops = list(dag.topological_op_nodes())
        nodes_seen = dict(zip(topo_ops, [False] * len(topo_ops)))
        for node in topo_ops:
            if isinstance(node.op, Gate) and \
                    len(node.qargs) == 1 and \
                    node.condition is None \
                    and not nodes_seen[node]:
                chain = [node]
                nodes_seen[node] = True
                successor = list(dag.successors(node))[0]  # has only one successor
                while successor.type == "op" and \
                        isinstance(successor.op, Gate) and \
                        len(successor.qargs) == 1 and \
                        successor.condition is None:
                    chain.append(successor)
                    nodes_seen[successor] = True
                    successor = list(dag.successors(successor))[0]
                chains.append(chain)

        # cannot collapse parameterized gates yet
        chains = self._split_chains_on_unknown_matrix(chains, dag)

        # collapse chains into a single unitary operator
        for chain in chains:
            if len(chain) == 1 and self.ignore_solo:
                continue
            matrix_chain = [gate.op.to_matrix() for gate in reversed(chain)]
            op = Operator(reduce(np.dot, matrix_chain))
            for node in chain[1:]:
                dag.remove_op_node(node)
            if op.equiv(np.eye(2)):
                dag.remove_op_node(chain[0])
            else:
                dag.substitute_node(chain[0], op.to_instruction(), inplace=True)

        return dag

    def _split_chains_on_unknown_matrix(self, chains, dag):
        """Finds chains containing parameterized gates or opaque gates or pulse
        gates (i.e. everything without a known matrix definition). Splits them into
        sequential chains excluding those gates. Additionally splits those that are
        in the basis and not recoverable by the one-qubit synthesizer.
        """
        # TODO: more elegant way of informing available synthesis basis
        _known_synthesis_basis = ['u', 'p', 'sx', 'r', 'rz', 'ry', 'rx', 'u1', 'u2', 'u3']

        def _unknown_matrix(op):
            if op.is_parameterized():
                return True
            try:
                mat = op.to_matrix()
            except CircuitError:
                mat = None
            return mat is None

        def _unrecoverable_op(op):
            return (self.basis_gates and op.name in self.basis_gates and
                    op.name not in _known_synthesis_basis)

        def _calibrated_op(node):
            if dag.calibrations and node.name in dag.calibrations:
                qubit = tuple([node.qargs[0].index])
                params = tuple(node.op.params)
                if (qubit, params) in dag.calibrations[node.name]:
                    return True
            return False

        out = []
        for chain in chains:
            groups = groupby(chain,
                             lambda x: (_unknown_matrix(x.op) or
                                        _unrecoverable_op(x.op) or
                                        _calibrated_op(x))
                             )

            for group_is_opaque, gates in groups:
                if not group_is_opaque:
                    out.append(list(gates))
        return out
