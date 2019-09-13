# -*- coding: utf-8 -*-

"""Pass for hoare circuit optimization.
"""
from qiskit.transpiler.basepasses import TransformationPass
from z3 import And, Not, Implies, Solver, Bool


class HoareOptimizer(TransformationPass):
    """ The inner workings of this are detailed in:
        https://arxiv.org/abs/1810.00375
    """

    def __init__(self):
        self.solver = Solver()
        self.variables = dict()
	self.gatenum = dict()

    def _gen_variable(self, qb):
        """ After each gate generate a new unique variable name for each of the
            qubits, using scheme: 'q[id]_[gatenum]', e.g. q1_0 -> q1_1 -> q1_2,
                                                          q2_0 -> q2_1
        """
        str = "q"+qb +"_"+self.gatenum[qb]
        self.gatenum[qb] += 1
	return Bool(str)

    def _initialize(self, dag):
        """ create boolean variables for each qubit and apply qb == 0 condition
        """
        pass

    def _add_postconditions(self, gate):
        """ create boolean variables for each qubit the gate is applied to
            and apply the relevant post conditions
        """
        pass

    def _traverse_dag(self, dag):
        """ traverse DAG
            - initialize boolean variables for each entry node
            - traverse gates:
                for each gate check: if any control is 0
                                     if triviality conditions are satisfied
                if yes remove gate from dag
                apply post conditions of gate
        """
        pass

    def run(self, dag):
        """
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        self._initialize(dag)
        self._traverse_dag(dag)

        return dag
