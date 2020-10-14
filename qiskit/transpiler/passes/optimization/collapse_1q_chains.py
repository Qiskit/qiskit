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


class Collapse1qChains(TransformationPass):
    """Collapse every chain of single-qubit gates into a 2x2 matrix operator.

    The resulting operators can be synthesized and optimized over the desired
    basis by invoking some follow-on passes.

    If the chain evaluates to identity (e.g. U(0,0,0)), this pass simply
    collapses the chain to none.
    """
    def __init__(self, ignore_solo=False):
        """
        Args:
            ignore_solo (bool): If True, all solo gates (chains of length one)
                are left untouched (faster). Otherwise, each will be converted
                to a 2x2 matrix operator.
        """
        self.ignore_solo = ignore_solo
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
        chains = _split_chains_on_unknown_def(chains)

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


def _split_chains_on_unknown_def(chains):
    """Finds chains containing parameterized gates or opaque gates (i.e. gates
    without a known matrix definition, e.g. pulse gates). Splits them into
    sequential chains excluding those gates.
    """
    out = []
    for chain in chains:
        groups = groupby(chain, lambda x: (x.op.is_parameterized() or
                                           x.op.definition is None))

        for group_is_parameterized, gates in groups:
            if not group_is_parameterized:
                out.append(list(gates))
    return out
