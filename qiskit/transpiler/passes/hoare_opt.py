# -*- coding: utf-8 -*-

""" Pass for hoare logic circuit optimization. """

import sys
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import ControlledGate
try:
    from z3 import And, Or, Not, Implies, Solver, Bool, unsat
except ModuleNotFoundError:
    print("Please install Z3 via 'pip install z3-solver'.")
    sys.exit(1)
import qiskit.transpiler.passes._gate_extension


class HoareOptimizer(TransformationPass):
    """ This is a transpiler pass using hoare logic circuit optimization.
        The inner workings of this are detailed in:
        https://arxiv.org/abs/1810.00375
    """
    def __init__(self, size=10):
        super().__init__()
        self.solver = Solver()
        self.variables = dict()
        self.gatenum = dict()
        self.gatecache = dict()
        self.varnum = dict()
        self.size = size        # gate cache size

    def _gen_variable(self, qb_id):
        """ After each gate generate a new unique variable name for each of the
            qubits, using scheme: 'q[id]_[gatenum]', e.g. q1_0 -> q1_1 -> q1_2,
                                                          q2_0 -> q2_1
        """
        varname = "q" + str(qb_id) + "_" + str(self.gatenum[qb_id])
        var = Bool(varname)
        self.gatenum[qb_id] += 1
        self.variables[qb_id].append(var)
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
                Implies(ctrl_ones, gate._postconditions(*(trgtvar + new_vars)))
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
            triv_cond = gate._trivial_if(*trgtvar)
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
            for each gate check: if any control is 0, or
                                 if triviality conditions are satisfied
            if yes remove gate from dag
            apply postconditions of gate
        """
        for node in dag.topological_op_nodes():
            gate = node.op
            _, ctrlvar, trgtqb, trgtvar = self._seperate_ctrl_trgt(node)

            ctrl_ones = And(*ctrlvar)

            trivial = self._test_gate(gate, ctrl_ones, trgtvar)
            if trivial:
                dag.remove_op_node(node)
            elif self.size > 1 and not trivial:
                for qb in node.qargs:
                    self.gatecache[qb.index].append(node)
                    self.varnum[qb.index][node] = self.gatenum[qb.index]-1
                for qb in node.qargs:
                    if len(self.gatecache[qb.index]) >= self.size:
                        self._multigate_opt(dag, qb.index)

            self._add_postconditions(gate, ctrl_ones, trgtqb, trgtvar)

    def _target_successive_seq(self, dag, qb_id):
        """ gates are target successive if they have the same set of target
            qubits and follow each other immediately on these target qubits
            (consider sequences of length 2 for now)
        """
        seqs = []
        for i in range(len(self.gatecache[qb_id])-1):
            append = True
            node1 = self.gatecache[qb_id][i]
            node2 = self.gatecache[qb_id][i+1]
            trgtqb1 = self._seperate_ctrl_trgt(node1)[2]
            trgtqb2 = self._seperate_ctrl_trgt(node2)[2]

            if trgtqb1 != trgtqb2:
                continue
            try:
                for qb in trgtqb1:
                    idx = self.gatecache[qb.index].index(node1)
                    if self.gatecache[qb.index][idx+1] is not node2:
                        append = False
            except (IndexError, ValueError):
                continue

            if append:
                seqs.append([node1, node2])

        return seqs

    def _is_identity(self, sequence):
        """ determine whether the sequence of gates combines to the idendity
            (consider sequences of length 2 for now)
        """
        assert len(sequence) == 2
        # DOESN'T WORK FOR GATES WITH DIFFERENT CONTROL QUBITS YET
        return isinstance(sequence[0].op, type(sequence[1].op.inverse()))

    def _seq_as_one(self, sequence):
        """ use z3 solver to determine if the gates in the sequence are either
            all executed or none of them are executed, based on control qubits
            (consider sequences of length 2 for now)
        """
        assert len(sequence) == 2
        ctrlvar1 = self._seperate_ctrl_trgt(sequence[0])[1]
        ctrlvar2 = self._seperate_ctrl_trgt(sequence[1])[1]

        self.solver.push()
        self.solver.add(
            Or(
                And(And(*ctrlvar1), Not(And(*ctrlvar2))),
                And(Not(And(*ctrlvar1)), And(*ctrlvar2))
            )
        )
        res = self.solver.check() == unsat
        self.solver.pop()

        return res

    def _multigate_opt(self, dag, qb_id, max_idx=None, dnt_rec=None):
        """
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
            qb_id (int): qubit id whose gate cache is to be optimized
            max_idx (int): a value indicates a recursive call, optimize
                           and remove gates up to this point in the cache
            dnt_rec ([int]): don't recurse on these qubit caches (again)
        """
        if len(self.gatecache[qb_id]) == 0:
            return

        # try to optimize this qubit's pipeline
        for seq in self._target_successive_seq(dag, qb_id):
            if self._is_identity(seq) and self._seq_as_one(seq):
                for node in seq:
                    dag.remove_op_node(node)
                    # if recursive call, gate will be removed further down
                    if max_idx is None:
                        for qb in node.qargs:
                            self.gatecache[qb.index].remove(node)

        if len(self.gatecache[qb_id]) < self.size and max_idx is None:
            # unless in a rec call, we are done if the cache isn't full
            return
        elif max_idx is None:
            # need to remove at least one gate from cache, so remove oldest
            max_idx = 0
            dnt_rec = set()
            dnt_rec.add(qb_id)
            gates_tbr = [self.gatecache[qb_id][0]]
        else:
            # need to remove all gates up to max_idx
            gates_tbr = self.gatecache[qb_id][max_idx::-1]

        for node in gates_tbr:
            # for rec call, only look at qubits that haven't been optimized yet
            new_qb = [x.index for x in node.qargs if x.index not in dnt_rec]
            dnt_rec.update(new_qb)
            for qb in new_qb:
                idx = self.gatecache[qb].index(node)
                # recursive chain to optimize all gates in this qubit's cache
                self._multigate_opt(dag, qb, max_idx=idx, dnt_rec=dnt_rec)
        # truncate gatecache for this qubit to after above gate
        self.gatecache[qb_id] = self.gatecache[qb_id][max_idx+1:]

    def _seperate_ctrl_trgt(self, node):
        """ Get the target qubits and control qubits if available,
            as well as their respective z3 variables.
        """
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
        self._traverse_dag(dag)
        if self.size > 1:
            for qb in dag.qubits():
                self._multigate_opt(dag, qb.index)
        return dag
