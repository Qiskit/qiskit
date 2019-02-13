# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A core algorithm for 'FlexlayerSwap'.
"""
import collections
import copy
import logging
import pprint

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import SwapGate
from qiskit.mapper import MapperError, CouplingMap, Layout
from .ancestors import Ancestors
from .dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)


class FlexlayerHeuristics:

    def __init__(self,
                 qc: QuantumCircuit,
                 dependency_graph: DependencyGraph,
                 coupling: CouplingMap,
                 initial_layout: Layout,
                 lookahead_depth: int = 5,
                 decay_rate: float = 0.5):

        self._qc = qc

        if initial_layout is None:
            raise MapperError("FlexlayerHeuristics requires initial_layout")

        self._dg = dependency_graph
        self._coupling = coupling
        self._initial_layout = copy.deepcopy(initial_layout)

        self._qubit_count_validity_check()
        if len(initial_layout.get_virtual_bits()) != len(initial_layout.get_physical_bits()):
            raise MapperError("FlexlayerHeuristics assumes #virtual-qubits == #physical-qubits")
        # self._add_ancilla_qubits()

        self._lookahead_depth = lookahead_depth
        self._decay_rate = decay_rate

        self._ancestors = Ancestors(self._dg._graph)  # for speed up

    def search(self) -> (DAGCircuit, Layout):
        """

        Returns:
            Mapped physical circuit (DAGCircuit) and initial qubit layout (Layout)
        """
        qreg = QuantumRegister(self._coupling.size(), name='q')
        new_dag = self._create_empty_dagcircuit(qreg)

        # initialize blocking gates
        blocking_gates = self._dg.head_gates()
        # initialize layout
        layout = copy.deepcopy(self._initial_layout)
        logger.debug("initial_layout = %s", pprint.pformat(layout))

        MAX_N_ITERATION = self._coupling.size() * (self._dg.n_nodes() ** 2)
        for k in range(1, MAX_N_ITERATION + 1):  # escape infinite loop
            logger.debug("iteration %d", k)
            logger.debug("layout=%s", pprint.pformat(layout))

            # update blocking gates
            blocking_gates, dones = self._next_blocking_gates_from(blocking_gates, layout=layout)
            logger.debug("#blocking_gates = %s", pprint.pformat(blocking_gates))
            logger.debug("#done_gates = %d", len(dones))

            if len(dones) > 0:
                for g in dones:
                    new_dag.apply_operation_back(self._dg.gate(g, layout, qreg))

            if len(blocking_gates) == 0:
                break

            ece = EdgeCostEstimator(gates=blocking_gates,
                                    layout=layout,
                                    coupling=self._coupling,
                                    dg=self._dg,
                                    max_depth=self._lookahead_depth)

            costs = [ece.cost(e, alpha=self._decay_rate) for e in ece.cand_edges]

            min_cost, e = min(zip(costs, ece.cand_edges))

            if min_cost.immediate_cost < 0:
                # usual case
                logger.debug("swap min-cost edge = %s", pprint.pformat(e))
                e = self._fix_swap_direction(e)
                new_dag.apply_operation_back(SwapGate(qreg[e[0]], qreg[e[1]]))
                # update layout
                layout.swap(e[0], e[1])
            else:
                # special case necessary to avoid cyclic swaps
                logger.debug("cannot reduce total path length -> resolve a single path")
                focus_gates = self._find_focus_gates(gates=blocking_gates,
                                                     layout=layout)

                MAX_N_INNER_LOOPS = len(focus_gates) * self._coupling.size()
                for kk in range(MAX_N_INNER_LOOPS):  # escape infinite loop
                    ece = EdgeCostEstimator(gates=focus_gates,
                                            layout=layout,
                                            coupling=self._coupling,
                                            dg=self._dg,
                                            max_depth=self._lookahead_depth)

                    costs = [ece.cost(e,
                                      priority=['immediate_cost', 'lookahead_cost'],
                                      alpha=self._decay_rate)
                             for e in ece.cand_edges]

                    min_cost, e = min(zip(costs, ece.cand_edges))

                    e = self._fix_swap_direction(e)
                    new_dag.apply_operation_back(SwapGate(qreg[e[0]], qreg[e[1]]))
                    layout.swap(e[0], e[1])

                    logger.debug("%d-th inner iter. add a swap (%d, %d)", kk, e[0], e[1])
                    logger.debug("resolved layout = %s", pprint.pformat(sorted(layout.items())))

                    dones = self._find_done_gates(blocking_gates, layout=layout)
                    if len(dones) > 0:
                        break

                if kk == MAX_N_INNER_LOOPS:
                    raise MapperError("Unknown error (maybe infinite inner loop)")  # bug

            if k == MAX_N_ITERATION:
                raise MapperError("UnknowError: #iteration reached MAX_N_ITERATION")

        reslayout = self._initial_layout
        # resqc = dag_to_circuit(new_dag)
        # resqc, reslayout = remove_head_swaps(resqc, self._initial_layout)

        return new_dag, reslayout

    def _next_blocking_gates_from(self, blocking_gates, layout):
        """next blocking gates from blocking_gates for layout with finding applicable gates (dones)
        """
        leadings = set(blocking_gates)  # new leading gates
        dones = []
        new_dones = self._find_done_gates(leadings, layout)
        while len(new_dones) > 0:
            dones.extend(new_dones)
            leadings = self._update_leading_gates(leadings, new_dones)
            new_dones = self._find_done_gates(leadings, layout)

        assert len(dones) == len(set(dones))
        # dones must be list (order is essential!)

        return frozenset(leadings), dones

    def _find_done_gates(self, blocking_gates, layout):
        dones = []
        for n in blocking_gates:
            qargs = self._dg.qargs(n)
            if self._dg.gate_name(n) == "barrier":
                dones.append(n)
            elif len(qargs) == 1:  # 1-qubit gate (including measure)
                dones.append(n)
            elif len(qargs) == 2:  # CNOT(2-qubit gate)
                dist = self._coupling.distance(layout[qargs[0]],
                                               layout[qargs[1]])
                if dist == 1:
                    dones.append(n)
            else:
                raise MapperError("DG contains unknown >2 qubit gates")

        return dones

    def ancestors(self, n) -> set:
        return self._ancestors.ancestors(n)

    def _update_leading_gates(self, leading_gates, dones):
        news = set(leading_gates)
        for n in dones:
            news |= set(self._dg.gr_successors(n))

        news -= set(dones)

        rmlist = []
        for n in news:
            if len(news & self.ancestors(n)) > 0:
                rmlist.append(n)

        news -= set(rmlist)

        return news

    def _fix_swap_direction(self, e):
        if e in self._coupling.get_edges():
            return e
        else:
            return e[1], e[0]

    def _find_focus_gates(self, gates, layout):
        if len(gates) <= 1:
            logger.debug("_find_focus_gates: %d <= 1 gates" % len(gates))
            return gates

        paths = {g: self._path_of(g, layout) for g in gates}
        gce = GateCostEstimator(gates, paths)

        costs = [gce.cost(g) for g in gce.gates]

        _, best_gates = min(zip(costs, gce.gates))

        return [best_gates]

    def _path_of(self, gate, layout):
        """
        gate: two-qubit gate
        """
        qargs = self._dg.qargs(gate)
        if len(qargs) != 2:
            raise MapperError("gate must be a two-qubit gate")
        s, t = layout[qargs[0]], layout[qargs[1]]
        path = self._coupling.shortest_undirected_path(s, t)
        return path

    def _qubit_count_validity_check(self):
        virtual_qubits = list(self._dg.qubits)
        physical_qubits = list(self._coupling.physical_qubits)

        if len(virtual_qubits) > len(physical_qubits):
            raise MapperError("Not enough qubits in _coupling")

        for q in self._initial_layout.get_physical_bits():
            if q not in physical_qubits:
                raise MapperError("%s isn't in _coupling but in initial_layout" % pprint.pformat(q))

    def _add_ancilla_qubits(self):
        virtual_qubits = sorted(self._dg.qubits)
        physical_qubits = sorted(self._coupling.physical_qubits)
        for b in self._initial_layout.get_virtual_bits():
            if b not in virtual_qubits:
                del self._initial_layout[b]
                logger.info("remove unused qubit in initial_layout %s" % pprint.pformat(b))

        ancilla_qubits = set(physical_qubits) - set(self._initial_layout.get_physical_bits().keys())

        # add ancilla qubits
        if len(ancilla_qubits) > 0:
            anc_qreg = QuantumRegister(len(ancilla_qubits), name="ancilla")
            for i, aq in enumerate(ancilla_qubits):
                virtual_qubits.append((anc_qreg, i))
                self._initial_layout[(anc_qreg, i)] = aq

        assert len(physical_qubits) == len(virtual_qubits), \
            "physical_qubits=%d, virtual_qubits=%d" % (len(physical_qubits), len(virtual_qubits))

    def _create_empty_dagcircuit(self, physical_qreg):
        new_dag = DAGCircuit()
        new_dag.add_qreg(physical_qreg)
        for creg in self._qc.cregs:
            new_dag.add_creg(creg)
        # new_dag.add_basis_element('swap', 2, 0, 0)
        # for name, data in self._qc.definitions.items():
        #     new_dag.add_basis_element(name, data["n_bits"], 0, data["n_args"])
        #     new_dag.add_gate_data(name, data)
        return new_dag


class GateCostEstimator:

    def __init__(self, gates, paths):
        self.gates = gates
        self.paths = paths

    def cost(self, gate, priority=None):
        """estimate cost if the gate pair (specified by gate_idx_pair) are resolved first
        """
        if priority is None:
            priority = ['dependent_cost']

        Cost = collections.namedtuple('Cost', priority)

        pi = self.paths[gate]

        dc = 0  # dependent_cost
        for j in self.gates:
            pj = self.paths[j]
            if pj[0] in pi or pj[-1] in pi:
                dc += 1

        return Cost(dependent_cost=dc)


class EdgeCostEstimator:

    def __init__(self, gates, layout, coupling, dg, max_depth=10):
        self.cand_edges = []
        for gate in gates:
            qargs = dg.qargs(gate)
            if len(qargs) != 2:
                raise MapperError("EdgeCostEstimator is only for two-qubit gates")
            s, t = layout[qargs[0]], layout[qargs[1]]
            for v in coupling.undirected_neighbors(s):
                if coupling.distance(s, t) > coupling.distance(v, t):
                    self.cand_edges.append((s, v))
            for u in coupling.undirected_neighbors(t):
                if coupling.distance(s, t) > coupling.distance(s, u):
                    self.cand_edges.append((u, t))

        self.care_ends_list = self._construct_care_ends_list(gates, layout, dg, max_depth)

        self.coupling = coupling

    def cost(self, e, priority=None, alpha=0.5):
        """
        estimate cost if the edge e is swapped
        Args:
            e: edge whose cost is estimated as pair of physical qubits, ex. (('q', 1), ('q', 2))
            priority: costs are sorted by this lexicographic order
            alpha: discount rate for cost of post layer gates
        """
        if priority is None:
            priority = ['lookahead_cost', 'immediate_cost']

        Cost = collections.namedtuple('Cost', priority)
        lc = 0  # lookahead_cost
        ic = 0  # immediate_cost
        for (s, t), d in self.care_ends_list:
            inc = 0
            if (s in e) ^ (t in e):  # xor: either node of e is s or t of p (path from s to t)
                if s == e[0]:
                    inc = self._after_swap_cost(e[0], e[1], t)
                elif s == e[1]:
                    inc = self._after_swap_cost(e[1], e[0], t)
                elif t == e[0]:
                    inc = self._after_swap_cost(e[0], e[1], s)
                elif t == e[1]:
                    inc = self._after_swap_cost(e[1], e[0], s)
                else:
                    raise MapperError("Unknown error")

            lc += inc * (alpha ** d)
            if d == 0:  # current layer
                ic += inc

        return Cost(lookahead_cost=lc, immediate_cost=ic)

    def _construct_care_ends_list(self, gates, layout, dg, max_depth):
        nodes = set(gates)
        dic = collections.defaultdict(int)  # approximate max path length from gates to the key gate
        for d in range(max_depth):
            nexts = {s for n in nodes for s in dg.gr_successors(n)}
            for n in nexts:
                if len(dg.qargs(n)) == 2:
                    dic[n] = d + 1
            nodes = nexts

        care_ends_list = []
        for g in gates:
            qargs = dg.qargs(g)
            s, t = layout[qargs[0]], layout[qargs[1]]
            care_ends_list.append(((s, t), 0))

        for g, d in dic.items():
            if d <= max_depth:
                qargs = dg.qargs(g)
                s, t = layout[qargs[0]], layout[qargs[1]]
                care_ends_list.append(((s, t), d))

        return care_ends_list

    def _after_swap_cost(self, s1, s2, t):
        """return distance difference in switching path s1->t to path s2->t
        for the coupling as undirected graph
        """
        d1 = self.coupling.distance(s1, t)
        d2 = self.coupling.distance(s2, t)
        if abs(d1 - d2) > 1:
            raise MapperError("Invalid shortest paths on coupling graph")
        return d2 - d1


def _qargs(gate):
    return [ridx for (reg, ridx) in gate.qargs]


def remove_head_swaps(qc: QuantumCircuit,
                      initial_layout: Layout) -> (QuantumCircuit, Layout):
    """remove unnecessary swap gates from qc by changing initial_layout
    assume all of the gates in qc are expanded to one-qubit gate, cx, swap
    """

    rev_new_layout = {v: k for k, v in initial_layout.items()}
    cx_seen = collections.defaultdict(bool)  # phisical qubit -> seen first cx or not
    to_be_removed = []
    for i, gate in enumerate(qc.data):
        if len(gate.qargs) > 1:
            qargs = _qargs(gate)
            if gate.name == "swap":
                if (not cx_seen[qargs[0]]) and (not cx_seen[qargs[1]]):
                    # swap before the first cx -> update rev_layout and do not output swap to qasm
                    rev_new_layout[qargs[0]], rev_new_layout[qargs[1]] = rev_new_layout[qargs[1]], \
                                                                         rev_new_layout[qargs[0]]
                    to_be_removed.append(i)
                else:
                    # must change flag! because left swap = cx
                    cx_seen[qargs[0]] = True
                    cx_seen[qargs[1]] = True
            else:
                for qarg in qargs:
                    cx_seen[qarg] = True

    logger.debug("remove_head_swaps: n_removed = %d" % len(to_be_removed))

    resqc = copy.deepcopy(qc)
    new_initial_layout = {v: k for k, v in rev_new_layout.items()}
    new_layout = new_initial_layout
    rev_org_layout = {v: k for k, v in initial_layout.items()}

    if len(to_be_removed) > 0:
        qregs = resqc.qregs
        for i, gate in enumerate(resqc.data[:1 + to_be_removed[-1]]):
            qargs = _qargs(gate)
            if gate.name == "swap":
                rev_org_layout[qargs[0]], rev_org_layout[qargs[1]] = rev_org_layout[qargs[1]], \
                                                                     rev_org_layout[qargs[0]]
                if i not in to_be_removed:
                    rev_new_layout[qargs[0]], rev_new_layout[qargs[1]] = rev_new_layout[qargs[1]], \
                                                                         rev_new_layout[qargs[0]]
                    new_layout = {v: k for k, v in rev_new_layout.items()}
            else:
                _change_qargs(gate, rev_org_layout, new_layout, qregs)

        resqc.data = [g for i, g in enumerate(resqc.data) if i not in to_be_removed]

    return resqc, new_initial_layout


def _change_qargs(gate: Gate, rev_org_layout: dict, new_layout: dict, qregs: dict):
    if gate.name == "measure":
        raise MapperError("_change_qargs() is not applicable to measure gate")

    new_qarg = []
    for qubit in _qargs(gate):
        new_qubit = new_layout[rev_org_layout[qubit]]
        new_qarg.append((qregs[new_qubit[0]], new_qubit[1]))

    gate.arg = new_qarg
