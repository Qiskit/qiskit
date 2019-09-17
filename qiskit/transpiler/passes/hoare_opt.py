# -*- coding: utf-8 -*-

"""Pass for hoare circuit optimization.
"""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import ControlledGate
from qiskit.dagcircuit import DAGCircuit
from z3 import And, Or, Not, Implies, Solver, Bool, sat, unsat
import qiskit.transpiler.passes._gate_extension


class HoareOptimizer(TransformationPass):
    """ The inner workings of this are detailed in:
        https://arxiv.org/abs/1810.00375
    """

    def __init__(self, l=10):
        super().__init__()
        self.solver = Solver()
        self.variables = dict()
        self.gatenum = dict()
        self.gatecache = dict()
        self.varnum = dict()
        self.l = l

    def _gen_variable(self, qb):
        """ After each gate generate a new unique variable name for each of the
            qubits, using scheme: 'q[id]_[gatenum]', e.g. q1_0 -> q1_1 -> q1_2,
                                                          q2_0 -> q2_1
        """
        varname = "q" + str(qb) + "_" + str(self.gatenum[qb])
        var = Bool(varname)
        self.gatenum[qb] += 1
        self.variables[qb].append(var)
        return var

    def _initialize(self, dag):
        """ create boolean variables for each qubit and apply qb == 0 condition
        """
        for qb in dag.qubits():
            self.gatenum[qb.index] = 0
            self.variables[qb.index] = []
            self.gatecache[qb.index] = []
            self.varnum[qb.index] = dict()
            x = self._gen_variable(qb.index)
            self.solver.add(x == False)

    def _add_postconditions(self, gate, ctrl_ones, trgtqb, trgtvar):
        """ create boolean variables for each qubit the gate is applied to
            and apply the relevant post conditions.
            a gate rotating out of the z-basis will not have any valid
            post-conditions, in which case the qubit state is unknown
        """
        new_vars = []
        for qb in trgtqb:
            new_vars.append(self._gen_variable(qb.index))

        try:
            self.solver.add(
                Implies(ctrl_ones, gate.postconditions(*(trgtvar + new_vars)))
            )
        except AttributeError:
            pass

        for i in range(len(trgtvar)):
            self.solver.add(
                Implies(Not(ctrl_ones), new_vars[i] == trgtvar[i])
            )

    def _test_gate(self, gate, ctrl_ones, trgtvar):
        """ use z3 sat solver to determine triviality of gate
        """
        trivial = False
        self.solver.push()

        try:
            triv_cond = gate.trivial_if(*trgtvar)
        except AttributeError:
            self.solver.add(ctrl_ones)
            trivial = self.solver.check() == unsat
        else:
            if isinstance(triv_cond, bool):
                if triv_cond and len(trgtvar) == 1:
                    self.solver.add(And(ctrl_ones, Not(trgtvar[0])))
                    s1 = self.solver.check() == unsat
                    self.solver.pop()
                    self.solver.push()
                    self.solver.add(And(ctrl_ones, trgtvar[0]))
                    s2 = self.solver.check() == unsat
                    trivial = s1 or s2
            else:
                self.solver.add(And(ctrl_ones, Not(triv_cond)))
                trivial = self.solver.check() == unsat

        self.solver.pop()
        return trivial

    def _traverse_dag(self, dag):
        """ traverse DAG
            for each gate check: if any control is 0
                                 if triviality conditions are satisfied
            if yes remove gate from dag
            apply postconditions of gate
        """
        new_dag = DAGCircuit()
        for l in dag.serial_layers():
            layer_dag = l['graph']
            nodes = layer_dag.gate_nodes()
            if len(nodes) != 1:
                continue
            node, gate = nodes[0], nodes[0].op

            var = self._seperate_ctrl_trgt(node)
            ctrlvar = var[1]
            trgtqb, trgtvar = var[2], var[3]

            ctrl_ones = And(*ctrlvar)

            trivial = self._test_gate(gate, ctrl_ones, trgtvar)
            if trivial:
                layer_dag.remove_op_node(node)
            elif self.l > 1:
                for qb in node.qargs:
                    self.gatecache[qb.index].append(node)
                    self.varnum[qb.index][node] = self.gatenum[qb.index]-1
                    if len(self.gatecache[qb.index]) >= self.l:
                        self._multigate_opt(qb.index)

            self._add_postconditions(gate, ctrl_ones, trgtqb, trgtvar)
            new_dag.extend_back(layer_dag)
        return new_dag

    def _target_successive_seq(self, dag, qb, max_idx):
        """ gates are target successive if they have the same set of target
            qubits and follow each other immediately on these target qubits
            (consider sequences of length 2 for now)
        """
        if max_idx is None:
            max_idx = len(self.gatecache[qb.index])-1
        if max_idx >= 1:
            seqs = []
            for i in range(1, max_idx+1):
                g1 = self.gatecache[qb.index][i-1]
                g2 = self.gatecache[qb.index][i]
                if g1.qargs == g2.qargs:
                    seqs.append([g1, g2])
            return seqs
        else:
            return []

    def _is_identity(self, sequence):
        """ determine whether the sequence of gates combines to the idendity
            (consider sequences of length 2 for now)
        """
        assert len(sequence) == 2
        return isinstance(sequence[0], type(sequence[1].inverse()))

    def _seq_as_one(self, sequence):
        """ use z3 solver to determine if the gates in the sequence are either
            all executed or none none of them executed, based on control qubits
            (consider sequences of length 2 for now)
        """
        assert len(sequence) == 2
        g1, g2 = sequence[0], sequence[1]
        v1, v2 = self._seperate_ctrl_trgt(g1), self._seperate_ctrl_trgt(g2)
        ctrlvar1, ctrlvar2 = v1[1], v2[1]

        self.solver.push()
        self.solver.add(
            Or(
                And(And(*ctrlvar1), Not(And(*ctrlvar2))),
                And(Not(And(*ctrlvar1)), And(*ctrlvar2))
            )
        )
        res = self.solver.check() == sat
        self.solver.pop()

        return res

    def _multigate_opt(self, dag, qb, rec=False, max_idx=None):
        """
        """
        rem = False
        for seq in self._target_successive_seq(dag, qb, max_idx):
            if self._is_identity(seq) and self._seq_as_one(seq):
                for node in seq:
                    dag.remove_op_node(node)
                    for qb in node.qargs:
                        self.gatecache[qb.index].remove(node)
                rem = True
        if not rem and not rec:
            # need to remove at least one gate from cache, so remove oldest
            first_gate = self.gatecache[qb.index][0]
            for qb in first_gate.qargs:
                idx = self.gatecache[qb.index].index(first_gate)
                # optimize first if older gates exist before removing
                if idx >= 1:
                    self._multigate_opt(dag, qb, rec=True, max_idx=idx)
                # for other qubits remove all up to & including above gate
                self.gatecache[qb.index] = self.gatecache[qb.index][idx+1:]

    def _seperate_ctrl_trgt(self, node):
        gate = node.op
        if isinstance(gate, ControlledGate):
            numctrl = gate.num_ctrl_qubits
        else:
            numctrl = 0
        ctrlqb = node.qargs[:numctrl]
        trgtqb = node.qargs[numctrl:]
        try:
            ctrlvar = [self.variables[qb.index][self.varnum[qb.index][node]]
                       for qb in ctrlqb]
            trgtvar = [self.variables[qb.index][self.varnum[qb.index][node]]
                       for qb in trgtqb]
        except KeyError:
            ctrlvar = [self.variables[qb.index][-1] for qb in ctrlqb]
            trgtvar = [self.variables[qb.index][-1] for qb in trgtqb]
        return (ctrlqb, ctrlvar, trgtqb, trgtvar)

    def run(self, dag):
        """
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        self._initialize(dag)
        new_dag = self._traverse_dag(dag)
        if self.l > 1:
            for qb in dag.qubits():
                self._multigate_opt(dag, qb)

        return new_dag
