# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Crosstalk mitigation through adaptive instruction scheduling.
The scheduling algorithm is described in:
Prakash Murali, David C. Mckay, Margaret Martonosi, Ali Javadi Abhari,
Software Mitigation of Crosstalk on Noisy Intermediate-Scale Quantum Computers,
in International Conference on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), 2020.
Please cite the paper if you use this pass.
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.extensions.standard import SwapGate
import networkx as nx
from z3 import *
from qiskit.extensions.standard import U1Gate, U2Gate, U3Gate, CnotGate
from qiskit.circuit import Measure
import math
from itertools import chain, combinations
import pprint
import operator
from qiskit.extensions.standard.barrier import Barrier

NUM_PREC = 10
TWOQ_XTALK_THRESH = 3
ONEQ_XTALK_THRESH = 2
DebugMode = False


class CrosstalkAdaptiveSchedule(TransformationPass):
    def __init__(self, backend_prop, crosstalk_prop, params):
        super().__init__()
        self.backend_prop = backend_prop
        self.crosstalk_prop = crosstalk_prop
        self.params = params
        self.gate_id = {}
        self.bp_u1_err = {}
        self.bp_u1_dur = {}
        self.bp_u2_err = {}
        self.bp_u2_dur = {}
        self.bp_u3_err = {}
        self.bp_u3_dur = {}
        self.bp_cx_err = {}
        self.bp_cx_dur = {}
        self.bp_t1_time = {}
        self.bp_t2_time = {}

        # Z3 variables
        self.gate_start_time = {}
        self.gate_duration = {}
        self.gate_fidelity = {}
        self.overlap_amounts = {}
        self.overlap_indicator = {}
        self.qubit_lifetime = {}

    def powerset(self, iterable):
        """
        Finds the set of all subsets of the given iterable
        This function is used to generate constraints for the Z3 optimization
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def parse_backend_properties(self):
        """
        This function assumes that T1, T2 times are in microseconds and
        gate times are in nanoseconds in backend.properties()
        """
        backend_prop = self.backend_prop
        qid = 0
        for qinfo in backend_prop.qubits:
            for item in qinfo:
                if item.name == "T1":
                    # Convert us to ns
                    self.bp_t1_time[qid] = int(item.value*1000)
                elif item.name == "T2":
                    self.bp_t2_time[qid] = int(item.value*1000)
            qid += 1
        for ginfo in backend_prop.gates:
            if ginfo.gate == 'u1':
                q0 = ginfo.qubits[0]
                for item in ginfo.parameters:
                    if item.name == 'gate_error':
                        if item.value == 1.0:
                            self.bp_u1_err[q0] = 0.999999
                        else:
                            self.bp_u1_err[q0] = round(item.value, NUM_PREC)
                    elif item.name == "gate_length":
                        self.bp_u1_dur[q0] = int(item.value)
            elif ginfo.gate == 'u2':
                q0 = ginfo.qubits[0]
                for item in ginfo.parameters:
                    if item.name == 'gate_error':
                        if item.value == 1.0:
                            self.bp_u2_err[q0] = 0.999999
                        else:
                            self.bp_u2_err[q0] = round(item.value, NUM_PREC)
                    elif item.name == "gate_length":
                        self.bp_u2_dur[q0] = int(item.value)
            elif ginfo.gate == 'u3':
                q0 = ginfo.qubits[0]
                for item in ginfo.parameters:
                    if item.name == 'gate_error':
                        if item.value == 1.0:
                            self.bp_u3_err[q0] = 0.999999
                        else:
                            self.bp_u3_err[q0] = round(item.value, NUM_PREC)
                    elif item.name == "gate_length":
                        self.bp_u3_dur[q0] = int(item.value)
            elif ginfo.gate == 'cx':
                q0 = ginfo.qubits[0]
                q1 = ginfo.qubits[1]
                r0 = min(q0, q1)
                r1 = max(q0, q1)
                for item in ginfo.parameters:
                    if item.name == 'gate_error':
                        if item.value == 1.0:
                            self.bp_cx_err[(r0, r1)] = 0.999999
                        else:
                            self.bp_cx_err[(r0, r1)] = round(item.value, NUM_PREC)
                    elif item.name == "gate_length":
                        self.bp_cx_dur[(r0, r1)] = int(item.value)

    def cx_tuple(self, gate):
        physical_q0 = gate.qargs[0].index
        physical_q1 = gate.qargs[1].index
        r0 = min(physical_q0, physical_q1)
        r1 = max(physical_q0, physical_q1)
        return (r0, r1)

    def singleq_tuple(self, gate):
        physical_q0 = gate.qargs[0].index
        tup = (physical_q0)
        return tup

    def gate_tuple(self, gate):
        if len(gate.qargs) == 2:
            return self.cx_tuple(gate)
        else:
            return self.singleq_tuple(gate)

    def assign_gate_id(self, dag):
        idx = 0
        for gate in dag.gate_nodes():
            self.gate_id[gate] = idx
            idx += 1

    def extract_dag_overlap_sets(self, dag):
        """
        Gate A, B are overlapping if
        A is neither a descendant nor an ancestor of B
        """
        self.dag_overlap_set = {}
        for gate in dag.twoQ_gates():
            s1 = [d for d in dag.descendants(gate)]
            s2 = [a for a in dag.ancestors(gate)]
            overlap_set = []
            for tmp_gate in dag.gate_nodes():
                if tmp_gate == gate:
                    continue
                if tmp_gate in s1:
                    continue
                if tmp_gate in s2:
                    continue
                overlap_set.append(tmp_gate)
            self.dag_overlap_set[gate] = overlap_set

    def is_significant_xtalk(self, gate1, gate2):
        """
        Given two conditational gate error rates
        check if there is high crosstalk by comparing with independent error rates.
        """
        gate1_tup = self.gate_tuple(gate1)
        if len(gate2.qargs) == 2:
            gate2_tup = self.gate_tuple(gate2)
            independent_err_g1 = self.bp_cx_err[gate1_tup]
            independent_err_g2 = self.bp_cx_err[gate2_tup]
            rg1 = self.crosstalk_prop[gate1_tup][gate2_tup]/independent_err_g1
            rg2 = self.crosstalk_prop[gate2_tup][gate1_tup]/independent_err_g2
            if rg1 > TWOQ_XTALK_THRESH or rg2 > TWOQ_XTALK_THRESH:
                return True
        else:
            gate2_tup = self.gate_tuple(gate2)
            independent_err_g1 = self.bp_cx_err[gate1_tup]
            rg1 = self.crosstalk_prop[gate1_tup][gate2_tup]/independent_err_g1
            if rg1 > ONEQ_XTALK_THRESH:
                return True
        return False

    def extract_crosstalk_relevant_sets(self):
        """
        Extract the set of program gates which potentially have crosstalk noise
        """
        self.xtalk_overlap_set = {}
        for g in self.dag_overlap_set:
            self.xtalk_overlap_set[g] = []
            tup_g = self.gate_tuple(g)
            if tup_g not in self.crosstalk_prop:
                continue
            for par_g in self.dag_overlap_set[g]:
                tup_par_g = self.gate_tuple(par_g)
                if tup_par_g in self.crosstalk_prop[tup_g]:
                    if self.is_significant_xtalk(g, par_g):
                        if par_g not in self.xtalk_overlap_set[g]:
                            self.xtalk_overlap_set[g].append(par_g)
        if DebugMode:
            for g in self.xtalk_overlap_set:
                print("\nOverlap:", self.gate_tuple(g), "gate_id:", self.gate_id[g])
                for par_g in self.xtalk_overlap_set[g]:
                    print(self.gate_tuple(par_g), "gate_id:", self.gate_id[par_g])

    def create_z3_vars(self):
        self.opt = Optimize()
        for gate in self.dag.gate_nodes():
            if isinstance(gate, Measure):
                continue
            if isinstance(gate, Barrier):
                continue
            t_var_name = 't_' + str(self.gate_id[gate])
            d_var_name = 'd_' + str(self.gate_id[gate])
            f_var_name = 'f_' + str(self.gate_id[gate])
            self.gate_start_time[gate] = Real(t_var_name)
            self.gate_duration[gate] = Real(d_var_name)
            self.gate_fidelity[gate] = Real(f_var_name)
        for gate in self.xtalk_overlap_set:
            self.overlap_indicator[gate] = {}
            self.overlap_amounts[gate] = {}
        for g1 in self.xtalk_overlap_set:
            for g2 in self.xtalk_overlap_set[g1]:
                if len(g2.qargs) == 2 and g1 in self.overlap_indicator[g2]:
                    self.overlap_indicator[g1][g2] = self.overlap_indicator[g2][g1]
                    self.overlap_amounts[g1][g2] = self.overlap_amounts[g2][g1]
                else:
                    # Indicator variable for overlap of g1 and g2
                    var_name1 = 'olp_ind_' + str(self.gate_id[g1]) + '_' + str(self.gate_id[g2])
                    self.overlap_indicator[g1][g2] = Bool(var_name1)
                    var_name2 = 'olp_amnt_' + str(self.gate_id[g1]) + '_' + str(self.gate_id[g2])
                    self.overlap_amounts[g1][g2] = Real(var_name2)
        active_qubits_list = []
        for gate in self.dag.gate_nodes():
            for q in gate.qargs:
                active_qubits_list.append(q.index)
        for active_qubit in list(set(active_qubits_list)):
            q_var_name = 'l_' + str(active_qubit)
            self.qubit_lifetime[active_qubit] = Real(q_var_name)

        meas_q = []
        for node in self.dag.op_nodes():
            if isinstance(node.op, Measure):
                meas_q.append(node.qargs[0].index)

        self.measured_qubits = list(set(self.input_measured_indices).union(set(meas_q)))
        self.measure_start = Real('meas_start')

    def basic_bounds(self):
        """
        Basic variable bounds for optimization
        """
        for gate in self.gate_start_time:
            self.opt.add(self.gate_start_time[gate] >= 0)
        for gate in self.gate_duration:
            q0 = gate.qargs[0].index
            if isinstance(gate.op, U1Gate):
                d = self.bp_u1_dur[q0]
            elif isinstance(gate.op, U2Gate):
                d = self.bp_u2_dur[q0]
            elif isinstance(gate.op, U3Gate):
                d = self.bp_u3_dur[q0]
            elif isinstance(gate.op, CnotGate):
                d = self.bp_cx_dur[self.cx_tuple(gate)]
            self.opt.add(self.gate_duration[gate] == d)

    def scheduling_constraints(self):
        """
        DAG scheduling constraints optimization
        Sets overlap indicator variables
        """
        for gate in self.gate_start_time:
            for dep_gate in self.dag.successors(gate):
                if not dep_gate.type == 'op':
                    continue
                if isinstance(dep_gate.op, Measure):
                    continue
                if isinstance(dep_gate.op, Barrier):
                    continue
                fin_g = self.gate_start_time[gate] + self.gate_duration[gate]
                self.opt.add(self.gate_start_time[dep_gate] > fin_g)
        for g1 in self.xtalk_overlap_set:
            for g2 in self.xtalk_overlap_set[g1]:
                if len(g2.qargs) == 2 and self.gate_id[g1] > self.gate_id[g2]:
                    # Symmetry breaking
                    continue
                s1 = self.gate_start_time[g1]
                f1 = s1 + self.gate_duration[g1]
                s2 = self.gate_start_time[g2]
                f2 = s2 + self.gate_duration[g2]
                # This constraint enforces full or zero overlap between two gates
                before = (f1 < s2)
                after = (f2 < s1)
                overlap1 = And(s2 <= s1, f1 <= f2)
                overlap2 = And(s1 <= s2, f2 <= f1)
                self.opt.add(Or(before, after, overlap1, overlap2))
                intervals_overlap = And(s2 <= f1, s1 <= f2)
                self.opt.add(self.overlap_indicator[g1][g2] == intervals_overlap)

    def fidelity_constraints(self):
        for gate in self.gate_start_time:
            q0 = gate.qargs[0].index
            no_xtalk = False
            if gate not in self.xtalk_overlap_set:
                no_xtalk = True
            elif not self.xtalk_overlap_set[gate]:
                no_xtalk = True
            if no_xtalk:
                if isinstance(gate.op, U1Gate):
                    f = math.log(1.0)
                elif isinstance(gate.op, U2Gate):
                    f = math.log(1.0 - self.bp_u2_err[q0])
                elif isinstance(gate.op, U3Gate):
                    f = math.log(1.0 - self.bp_u3_err[q0])
                elif isinstance(gate.op, CnotGate):
                    f = math.log(1.0 - self.bp_cx_err[self.cx_tuple(gate)])
                self.opt.add(self.gate_fidelity[gate] == round(f, NUM_PREC))
            else:
                comb = list(self.powerset(self.xtalk_overlap_set[gate]))
                xtalk_set = set(self.xtalk_overlap_set[gate])
                for item in comb:
                    on_set = item
                    off_set = [i for i in xtalk_set if i not in on_set]
                    clauses = []
                    for g in on_set:
                        clauses.append(self.overlap_indicator[gate][g])
                    for g in off_set:
                        clauses.append(Not(self.overlap_indicator[gate][g]))
                    err = 0
                    if len(on_set) == 0:
                        err = self.bp_cx_err[self.cx_tuple(gate)]
                    elif len(on_set) == 1:
                        on_gate = on_set[0]
                        err = self.crosstalk_prop[self.gate_tuple(gate)][self.gate_tuple(on_gate)]
                    else:
                        err_list = []
                        for on_gate in on_set:
                            tmp_prop = self.crosstalk_prop[self.gate_tuple(gate)]
                            err_list.append(tmp_prop[self.gate_tuple(on_gate)])
                        err = max(err_list)
                    if err == 1.0:
                        err = 0.999999
                    val = round(math.log(1.0 - err), NUM_PREC)
                    self.opt.add(Implies(And(*clauses), self.gate_fidelity[gate] == val))

    def coherence_constraints(self):
        self.last_gate_on_qubit = {}
        for gate in self.dag.topological_op_nodes():
            if isinstance(gate.op, Measure):
                continue
            if isinstance(gate.op, Barrier):
                continue
            if len(gate.qargs) == 1:
                q0 = gate.qargs[0].index
                self.last_gate_on_qubit[q0] = gate
            else:
                q0 = gate.qargs[0].index
                q1 = gate.qargs[1].index
                self.last_gate_on_qubit[q0] = gate
                self.last_gate_on_qubit[q1] = gate

        self.first_gate_on_qubit = {}
        for gate in self.dag.topological_op_nodes():
            if len(gate.qargs) == 1:
                q0 = gate.qargs[0].index
                if q0 not in self.first_gate_on_qubit:
                    self.first_gate_on_qubit[q0] = gate
            else:
                q0 = gate.qargs[0].index
                q1 = gate.qargs[1].index
                if q0 not in self.first_gate_on_qubit:
                    self.first_gate_on_qubit[q0] = gate
                if q1 not in self.first_gate_on_qubit:
                    self.first_gate_on_qubit[q1] = gate

        for q in self.last_gate_on_qubit:
            g_last = self.last_gate_on_qubit[q]
            g_first = self.first_gate_on_qubit[q]
            finish_time = self.gate_start_time[g_last] + self.gate_duration[g_last]
            start_time = self.gate_start_time[g_first]
            if q in self.measured_qubits:
                self.opt.add(self.measure_start >= finish_time)
                self.opt.add(self.qubit_lifetime[q] == self.measure_start - start_time)
            else:
                # All qubits get measured simultaneously whether or not they need a measurement
                self.opt.add(self.measure_start >= finish_time)
                self.opt.add(self.qubit_lifetime[q] == finish_time - start_time)

    def objective_function(self):
        self.fidelity_terms = [self.gate_fidelity[gate] for gate in self.gate_fidelity]
        self.coherence_terms = []
        for q in self.qubit_lifetime:
            val = -self.qubit_lifetime[q]/min(self.bp_t1_time[q], self.bp_t2_time[q])
            self.coherence_terms.append(val)

        all_terms = []
        for item in self.fidelity_terms:
            all_terms.append(self.weight_factor*item)
        for item in self.coherence_terms:
            all_terms.append((1-self.weight_factor)*item)
        self.opt.maximize(Sum(all_terms))

    def r2f(self, val):
        return float(val.as_decimal(16).rstrip('?'))

    def extract_solution(self):
        prec = 3
        self.m = self.opt.model()
        result = {}
        for g in self.gate_start_time:
            start = self.r2f(self.m[self.gate_start_time[g]])
            dur = self.r2f(self.m[self.gate_duration[g]])
            result[g] = (start, start+dur)
        return result

    def solve_optimization(self):
        """
        Setup and solve a Z3 optimization for finding the best schedule
        """
        self.create_z3_vars()
        self.basic_bounds()
        self.scheduling_constraints()
        self.fidelity_constraints()
        self.coherence_constraints()
        self.objective_function()
        # Solve step
        self.opt.check()
        # Extract the schedule computed by Z3
        result = self.extract_solution()
        return result

    def check_dag_dependancy(self, gate1, gate2):
        if gate2 in self.dag.descendants(gate1):
            return True
        else:
            return False

    def check_xtalk_dependency(self, t1, t2):
        g1 = t1[0]
        s1 = t1[1]
        f1 = t1[2]
        g2 = t2[0]
        s2 = t2[1]
        f2 = t2[2]
        # We don't consider single qubit xtalk
        if len(g1.qargs) == 1 and len(g2.qargs) == 1:
            return False, ()
        if s2 <= f1 and s1 <= f2:
            # Z3 says it's ok to overlap these gates,
            # so no xtalk dependency needs to be checked
            return False, ()
        else:
            # Assert because we are iterating in Z3 gate start time order,
            # so if two gates are not overlapping, then the second gate has to
            # start after the first gate finishes
            assert(s2 >= f1)
            # Not overlapping, but we care about this dependency
            if len(g1.qargs) == 2 and len(g2.qargs) == 2:
                if g2 in self.xtalk_overlap_set[g1]:
                    cx1 = self.cx_tuple(g1)
                    cx2 = self.cx_tuple(g2)
                    barrier = tuple(sorted([cx1[0], cx1[1], cx2[0], cx2[1]]))
                    return True, barrier
            elif len(g1.qargs) == 1 and len(g2.qargs) == 2:
                if g1 in self.xtalk_overlap_set[g2]:
                    singleq = self.gate_tuple(g1)
                    cx = self.cx_tuple(g2)
                    print(singleq, cx)
                    barrier = tuple(sorted([singleq, cx[0], cx[1]]))
                    return True, barrier
            elif len(g1.qargs) == 2 and len(g2.qargs) == 1:
                if g2 in self.xtalk_overlap_set[g1]:
                    singleq = self.gate_tuple(g2)
                    cx = self.cx_tuple(g1)
                    barrier = tuple(sorted([singleq, cx[0], cx[1]]))
                    return True, barrier
            else:
                # Not overlapping, and we don't care about xtalk between these two gates
                return False, ()
        return False, ()

    def filter_candidates(self, candidates, layer, layer_id, triplet):
        """
        For a gate G and layer L,
        L is a candidate layer for G if no gate in L has a DAG dependency with G,
        and if Z3 allows gates in L and G to overlap.
        """
        curr_gate = triplet[0]
        curr_start = triplet[1]
        curr_fin = triplet[2]
        for prev_triplet in layer:
            prev_gate = prev_triplet[0]
            prev_start = prev_triplet[1]
            prev_end = prev_triplet[2]
            is_dag_dep = self.check_dag_dependancy(prev_gate, curr_gate)
            is_xtalk_dep, _ = self.check_xtalk_dependency(prev_triplet, triplet)
            if is_dag_dep or is_xtalk_dep:
                # If there is a DAG dependency, we can't insert in any previous layer
                # If there is Xtalk dependency, we can (in general) insert in previous layers,
                # but since we are iterating in the order of gate start times,
                # we should only insert this gate in subsequent layers
                for i in range(layer_id+1):
                    if i in candidates:
                        candidates.remove(i)
            return candidates

    def find_layer(self, layers, triplet):
        candidates = [i for i in range(len(layers))]
        for i, layer in enumerate(layers):
            candidates = self.filter_candidates(candidates, layer, i, triplet)
        if len(candidates) == 0:
            return len(layers)
            # Open a new layer
        else:
            return max(candidates)
            # Latest acceptable layer, right-alignment

    def generate_barriers(self, dag, layers):
        """
        For each gate g, see if a barrier is required to serialize it with
        some previously processed gate
        """
        barriers = []
        for i, layer in enumerate(layers):
            barriers.append(set())
            if i == 0:
                continue
            for t2 in layer:
                for j in range(i):
                    prev_layer = layers[j]
                    for t1 in prev_layer:
                        is_dag_dep = self.check_dag_dependancy(t1[0], t2[0])
                        is_xtalk_dep, curr_barrier = self.check_xtalk_dependency(t1, t2)
                        if is_dag_dep:
                            # Don't insert a barrier since there is a DAG dependency
                            continue
                        if is_xtalk_dep:
                            # Insert a barrier for this layer
                            barriers[-1].add(curr_barrier)
        return barriers

    def create_updated_dag(self, layers, barriers):
        """
        Given a set of layers and barries, construct a new dag
        """
        new_dag = DAGCircuit()
        for qreg in self.dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in self.dag.cregs.values():
            new_dag.add_creg(creg)
        canonical_register = new_dag.qregs['q']
        for i, layer in enumerate(layers):
            curr_barriers = barriers[i]
            for b in curr_barriers:
                current_qregs = []
                for idx in b:
                    current_qregs.append(canonical_register[idx])
                new_dag.apply_operation_back(Barrier(len(b)), current_qregs, [])
            for triplet in layer:
                gate = triplet[0]
                new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)

        for node in self.dag.op_nodes():
            if isinstance(node.op, Measure):
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def enforce_schedule_on_dag(self, dag, input_gate_times):
        """
        Z3 outputs start times for each gate.
        Some gates need to be serialized to implement the Z3 schedule.
        This function inserts barriers to implement those serializations
        """
        gate_times = []
        for key in input_gate_times:
            gate_times.append((key, input_gate_times[key][0], input_gate_times[key][1]))
        # Sort gates by start time
        sorted_gate_times = sorted(gate_times, key=operator.itemgetter(1))
        layers = []
        # Construct a set of layers. Each layer has a set of gates that
        # are allowed to fire in parallel according to Z3
        for triplet in sorted_gate_times:
            layer_idx = self.find_layer(layers, triplet)
            if layer_idx == len(layers):
                layers.append([triplet])
            else:
                layers[layer_idx].append(triplet)
        # Insert barries if necessray to enforce the above layers
        barriers = self.generate_barriers(dag, layers)
        new_dag = self.create_updated_dag(layers, barriers)
        return new_dag

    def run(self, dag, weight_factor=0.5, measured_indices=[]):
        """
        Main scheduling function
        """
        # pre-processing steps
        self.weight_factor = weight_factor
        self.dag = dag
        self.input_measured_indices = measured_indices
        self.parse_backend_properties()

        # process input program
        self.assign_gate_id(self.dag)
        self.extract_dag_overlap_sets(self.dag)
        self.extract_crosstalk_relevant_sets()

        # setup and solve a Z3 optimization
        z3_result = self.solve_optimization()

        # post-process to insert barriers
        new_dag = self.enforce_schedule_on_dag(self.dag, z3_result)
        return new_dag
