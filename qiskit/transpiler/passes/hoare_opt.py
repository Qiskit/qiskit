# -*- coding: utf-8 -*-

"""Pass for hoare circuit optimization.
"""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import ControlledGate
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
        str = "q" + qb + "_" + self.gatenum[qb]
        self.gatenum[qb] += 1
        return Bool(str)

    def _initialize(self, dag):
        """ create boolean variables for each qubit and apply qb == 0 condition
        """
        for qb in dag.qubits():
            i = qb[1]  # id: Qubit(QuantumRegister(3, 'q'), 0)
            x = self._gen_variable(i)
            self.solver.add(x == 0)
            self.variables[i] = x

    def _add_postconditions(self, gate, ctrl_ones, trgtvar):
        """ create boolean variables for each qubit the gate is applied to
            and apply the relevant post conditions
        """
        pass

    def _test_gate(self, gate, ctrl_ones, trgtvar):
        pass

    def _traverse_dag(self, dag):
        """ traverse DAG
            for each gate check: if any control is 0
                                 if triviality conditions are satisfied
            if yes remove gate from dag
            apply postconditions of gate
        """
        for l in dag.serial_layers():
            nodes = l['graph'].gate_nodes()
            if len(nodes) != 1:
                continue
            gate = nodes[0]

            if isinstance(gate, ControlledGate):
                numctrl = gate.num_ctrl_qubits
            else:
                numctrl = 0
            ctrlqb = gate.qargs[:numctrl]
            trgtqb = gate.qargs[numctrl:]
            ctrlvar = [qb[1] for qb in ctrlqb]
            trgtvar = [qb[1] for qb in trgtqb]

            ctrl_ones = And(*ctrlvar)

            trivial = self._test_gate(gate, ctrl_ones, trgtvar)
            if trivial:
                dag.remove_op_node(gate)

            self._add_postconditions(gate, ctrl_ones, trgtvar)

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
