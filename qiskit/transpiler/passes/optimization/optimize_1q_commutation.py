# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Reduce 1Q gate complexity by commuting through 2Q gates and resynthesizing."""

from copy import copy
import logging

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, RZXGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
    Optimize1qGatesDecomposition,
)

logger = logging.getLogger(__name__)


commutation_table = {
    RZXGate: (["rz", "p"], ["x", "sx", "rx"]),
    CXGate: (["rz", "p"], ["x", "sx", "rx"]),
}
"""
Simple commutation rules: G belongs to commutation_table[barrier_type][qubit_preindex] when G
commutes with the indicated barrier on that qubit wire.

NOTE: Does not cover identities like
          (X (x) I) .   CX = CX .   (X (x) X) ,  (duplication)
          (U (x) I) . SWAP = SWAP . (I (x) U) .  (permutation)

NOTE: These rules are _symmetric_, so that they may be applied in reverse.
"""


class Optimize1qGatesSimpleCommutation(TransformationPass):
    """
    Optimizes 1Q gate strings interrupted by 2Q gates by commuting the components and re-
    synthesizing the results.  The commutation rules are stored in `commutation_table`.

    NOTE: In addition to those mentioned in `commutation_table`, this pass has some limitations:
          + Does not handle multiple commutations in a row without intermediate progress.
          + Can only commute into positions where there are pre-existing runs.
          + Does not exhaustively test all the different ways commuting gates can be assigned to
            either side of a barrier to try to find low-depth configurations.  (This is particularly
            evident if all the gates in a run commute with both the predecessor and the successor
            barriers.)
    """

    # NOTE: A run from `dag.collect_1q_runs` is always nonempty, so we sometimes use an empty list
    #       to signify the absence of a run.

    def __init__(self, basis=None, run_to_completion=False):
        """
        Args:
            basis (List[str]): See also `Optimize1qGatesDecomposition`.
            run_to_completion (bool): If `True`, this pass retries until it is unable to do any more
                work.  If `False`, it finds and performs one optimization, and for full optimization
                the user is obligated to re-call the pass until the output stabilizes.
        """
        super().__init__()

        self._basis = basis
        self._optimize1q = Optimize1qGatesDecomposition(basis)
        self._run_to_completion = run_to_completion

    @staticmethod
    def _find_adjoining_run(dag, runs, run, front=True):
        """
        Finds the run which abuts `run` from the front (or the rear if `front == False`), separated
        by a blocking node.

        Returns a pair of the abutting multi-qubit gate and the run which it separates from this
        one. The next run can be the empty list `[]` if it is absent.
        """
        edge_node = run[0] if front else run[-1]
        blocker = next(dag.predecessors(edge_node) if front else dag.successors(edge_node))
        possibilities = dag.predecessors(blocker) if front else dag.successors(blocker)

        adjoining_run = []
        for possibility in possibilities:
            if isinstance(possibility, DAGOpNode) and possibility.qargs == edge_node.qargs:
                adjoining_run = next((run for run in runs if possibility in run), [])
                break

        return (blocker, adjoining_run)

    @staticmethod
    def _commute_through(blocker, run, front=True):
        """
        Pulls `DAGOpNode`s from the front of `run` (or the back, if `front == False`) until it
        encounters a gate which does not commute with `blocker`.

        Returns a pair of lists whose concatenation is `run`.
        """

        if run == []:
            return [], []

        run_clone = copy(run)

        commuted = []
        preindex, commutation_rule = None, None
        if isinstance(blocker, DAGOpNode):
            preindex = None
            for i, q in enumerate(blocker.qargs):
                if q == run[0].qargs[0]:
                    preindex = i

            commutation_rule = None
            if (
                preindex is not None
                and isinstance(blocker, DAGOpNode)
                and type(blocker.op) in commutation_table
            ):
                commutation_rule = commutation_table[type(blocker.op)][preindex]

        if commutation_rule is not None:
            while run_clone != []:
                next_gate = run_clone[0] if front else run_clone[-1]
                if next_gate.name not in commutation_rule:
                    break
                if front:
                    commuted.append(next_gate)
                    del run_clone[0]
                else:
                    commuted.insert(0, next_gate)
                    del run_clone[-1]

        if front:
            assert commuted + run_clone == run
            return commuted, run_clone
        else:
            assert run_clone + commuted == run
            return run_clone, commuted

    def _resynthesize(self, new_run):
        """
        Synthesizes an efficient circuit from a sequence `new_run` of `DAGOpNode`s.

        NOTE: Returns None when resynthesis is not possible.
        """
        if len(new_run) == 0:
            return (), QuantumCircuit(1)

        return self._optimize1q._resynthesize_run(new_run)

    @staticmethod
    def _replace_subdag(dag, old_run, new_circ):
        """
        Replaces a nonempty sequence `old_run` of `DAGNode`s, assumed to be a complete chain in
        `dag`, with the circuit `new_circ`.
        """

        new_dag = circuit_to_dag(new_circ)
        node_map = dag.substitute_node_with_dag(old_run[0], new_dag)

        for node in old_run[1:]:
            dag.remove_op_node(node)

        spliced_run = [node_map[node._node_id] for node in new_dag.topological_op_nodes()]
        mov_list(old_run, spliced_run)

    def _step(self, dag):
        """
        Performs one full pass of optimization work.

        Returns True if `dag` changed, False if no work on `dag` was possible.
        """

        runs = dag.collect_1q_runs()
        did_work = False

        for run in runs:
            # identify the preceding blocking gates
            run_clone = copy(run)
            if run == []:
                continue

            # try to modify preceding_run
            preceding_blocker, preceding_run = self._find_adjoining_run(dag, runs, run)
            commuted_preceding = []
            if preceding_run != []:
                commuted_preceding, run_clone = self._commute_through(preceding_blocker, run_clone)

            # try to modify succeeding run
            succeeding_blocker, succeeding_run = self._find_adjoining_run(
                dag, runs, run, front=False
            )
            commuted_succeeding = []
            if succeeding_run != []:
                run_clone, commuted_succeeding = self._commute_through(
                    succeeding_blocker, run_clone, front=False
                )

            # re-synthesize
            new_preceding_basis, new_preceding_run = self._resynthesize(
                preceding_run + commuted_preceding
            )
            new_succeeding_basis, new_succeeding_run = self._resynthesize(
                commuted_succeeding + succeeding_run
            )
            new_basis, new_run = self._resynthesize(run_clone)

            # perform the replacement if it was indeed a good idea
            if self._optimize1q._substitution_checks(
                dag,
                (preceding_run or []) + run + (succeeding_run or []),
                (
                    (new_preceding_run or QuantumCircuit(1)).data
                    + (new_run or QuantumCircuit(1)).data
                    + (new_succeeding_run or QuantumCircuit(1)).data
                ),
                new_basis + new_preceding_basis + new_succeeding_basis,
            ):
                if preceding_run and new_preceding_run is not None:
                    self._replace_subdag(dag, preceding_run, new_preceding_run)
                if succeeding_run and new_succeeding_run is not None:
                    self._replace_subdag(dag, succeeding_run, new_succeeding_run)
                if new_run is not None:
                    self._replace_subdag(dag, run, new_run)
                did_work = True

        return did_work

    def run(self, dag):
        """
        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        # python doesn't support tail calls
        while True:
            did_work = self._step(dag)
            if not self._run_to_completion or not did_work:
                break

        return dag


def mov_list(destination, source):
    """
    Replace `destination` in-place with `source`.
    """

    while destination:
        del destination[0]
    destination += source
