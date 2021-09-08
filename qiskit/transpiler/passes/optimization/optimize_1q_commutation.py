from copy import copy
import logging
import warnings

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate, RZXGate, U3Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization import Optimize1qGatesDecomposition

logger = logging.getLogger(__name__)


commutation_table = {
    RZXGate:
        (['rz', 'p'],
         ['x', 'sx', 'rx']),
    CXGate:
        (['rz', 'p'],
         ['x', 'sx', 'rx']),
}
"""
Simple commutation rules: G belongs to commutation_table[barrier_type][qubit_preindex] when G
commutes with the indicated barrier on that qubit wire.

NOTE: Does not cover identities like
          (X (x) I) .   CX = CX .   (X (x) X) ,  (duplication)
          (U (x) I) . SWAP = SWAP . (I (x) U) .  (permutation)
          
NOTE: These rules are _symmetric_, so that they may be applied in reverse.
"""


class Optimize1QGatesSimpleCommutation(TransformationPass):
    """
    Optimizes 1Q gate strings interrupted by 2Q gates by commuting the components and re-
    synthesizing the results.
    """

    def __init__(self, basis=None):
        """
        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects of this
                pass, the basis is the set intersection between the `basis` parameter and the Euler
                basis.
        """
        super().__init__()
        
        self._basis = basis
        self._optimize1q = Optimize1qGatesDecomposition(basis)
        
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
            if possibility.qargs == edge_node.qargs:
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
        run_clone = copy(run)
        
        commuted = []
        preindex, commutation_rule = None, None
        if isinstance(blocker, DAGOpNode):
            preindex = None
            for i, q in enumerate(blocker.qargs):
                if q == run[0].qargs[0]:
                    preindex = i
            
            commutation_rule = None
            if (preindex is not None and 
                    isinstance(blocker, DAGOpNode) and
                    type(blocker.op) in commutation_table):
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
        """
        
        new_circuit = QuantumCircuit(1)
        for gate in new_run:
            new_circuit.append(gate.op, [0])
        return self._optimize1q(new_circuit)
    
    @staticmethod
    def _replace_subdag(dag, old_run, new_circ):
        """
        Replaces a nonempty sequence `old_run` of `DAGNode`s, assumed to be a complete chain in
        `dag`, with the circuit `new_circ`.
        """
        
        new_dag = circuit_to_dag(new_circ)
        dag.substitute_node_with_dag(old_run[0], new_dag)
        for node in old_run[1:]:
            dag.remove_op_node(node)
        
        return
        
    def _step(self, dag):
        """
        Performs one unit of optimization work.
        
        Returns True if `dag` changed, False if no work on `dag` was possible.
        """
        
        runs = dag.collect_1q_runs()
        for run in runs:  # N.B.: no particular traversal order
            # identify the preceding blocking gates
            run_clone = copy(run)
            if run == []:
                continue
            
            # try to modify preceding_run
            preceding_blocker, preceding_run = self._find_adjoining_run(
                dag, runs, run
            )
            commuted_preceding = []
            if preceding_run != []:
                commuted_preceding, run_clone = self._commute_through(
                    preceding_blocker, run_clone
                )
            
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
            new_preceding_run = self._resynthesize(preceding_run + commuted_preceding)
            new_succeeding_run = self._resynthesize(commuted_succeeding + succeeding_run)
            new_run = self._resynthesize(run_clone)
            
            # check whether this was actually a good idea
            original_depth = len(run) + len(preceding_run) + len(succeeding_run)
            new_depth = len(new_run) + len(new_preceding_run) + len(new_succeeding_run)
            
            # perform the replacement if it was indeed a good idea
            if original_depth > new_depth:
                did_work = True
                if preceding_run != []:
                    self._replace_subdag(dag, preceding_run, new_preceding_run)
                if succeeding_run != []:
                    self._replace_subdag(dag, succeeding_run, new_succeeding_run)
                self._replace_subdag(dag, run, new_run)
                return True
        
        return False

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
            if not did_work:
                break
        
        return dag
