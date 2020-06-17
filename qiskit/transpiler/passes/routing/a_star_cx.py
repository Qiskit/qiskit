# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import copy
from timeit import default_timer as timer

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.routing.algorithms.group_gates import group_gates
from qiskit.transpiler.passes.routing.cython.a_star_mapper import a_star_mapper
from qiskit.transpiler.passes.routing.algorithms import post_mapping_optimization
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister
from qiskit import QuantumCircuit
from qiskit.transpiler.coupling import CouplingMap


"""Pass for minimizing the number of CX gates.
"""

# Competition winning entry


def JAGcoupling_list2dict(couplinglist):
    """Convert coupling map list into dictionary.

    Example list format: [[0, 1], [0, 2], [1, 2]]
    Example dictionary format: {0: [1, 2], 1: [2]}

    We do not do any checking of the input.

    Return coupling map in dict format.
    """
    if not couplinglist:
        return None
    couplingdict = {}
    for pair in couplinglist:
        if pair[0] in couplingdict:
            couplingdict[pair[0]].append(pair[1])
        else:
            couplingdict[pair[0]] = [pair[1]]
    return couplingdict


class AStarCX(TransformationPass):
    """Find cheapest cost (local optimization with lookahead) for CX gates.
       Uses an admissable A* search technique to reduce (not "minimize" as it is not guaranteed to
       be optimal in all cases) the number of swaps (and, therefore, CX gates) to implement a given
       circuit under the constraints of a supplied coupling map.
       group_gates is used to "compress" (hide) single-qubit gates so that the algorithm can
       deal with CX gates exclusively.  Then the A* mapper (.pyx) is used to, layer by layer,
       select a good qubit mapping so as to reduce the number of FX gates used.  Finally, a
       post_mapping_optimization (KAK refactoring, introducing gates to deal with unidirectional
       connectivity in coupling map, etc.) are applied.  This is done for some number of iterations,
       as the topological sort that primes the order in which gates are added varies (run to run).
       Solutions are evaluated for cost and the best solution is retained and returned.
    """

    def __init__(self, coupling_map=None, gate_costs=None):
        super().__init__()
        self.coupling_map = coupling_map
        self.gate_costs = gate_costs

    def run(self, dag):
        """
        Runs one pass A*-based algorithm to minimize CX Gate count.
        Note: Currently *chained* to Group_Gates because that routine
        does not yet take/return a DAG.

        Args:
        dag (DAGCircuit): DAGCircuit object to be compiled.
        coupling_circuit (list): Coupling map for device topology.
                                 A coupling map of None corresponds an
                                 all-to-all connected topology.
        gate_costs (dict) : dictionary of gate names and costs.

        Returns:
            DAGCircuit: Transformed DAG.  A modified DAGCircuit object
            that satisfies an input coupling_map and has as low a gate_cost
            as possible.
        """

        if self.gate_costs is None:
            self.gate_costs = {
                "id": 0,
                "u1": 0,
                "measure": 0,
                "reset": 0,
                "barrier": 0,
                "u2": 1,
                "u3": 1,
                "U": 1,
                "cx": 10,
                "CX": 10,
            }
        compiled_dag = DAGCircuit()
        compiled_dag = copy.deepcopy(dag)

        # temporary circuit to add all used gates to the available gate set
        tmp_circuit = QuantumCircuit(2)
        tmp_circuit.cx(0, 1)
        tmp_circuit.u3(0.1, 0.4, 0.7, 0)
        tmp_circuit.u2(0.1, 0.4, 0)
        tmp_circuit.u1(0.1, 0)

        # prepare empty circuit for the result
        empty_dag = DAGCircuit()
        coupling = CouplingMap(self.coupling_map)

        # Note: Works with a single register named 'q' (not with 'q0' for example)
        empty_dag.add_qreg(QuantumRegister(coupling.size(), "q"))
        for k, v in sorted(compiled_dag.cregs.items()):
            empty_dag.add_creg(ClassicalRegister(v, k))
        start = timer()
        empty_dag.basis = compiled_dag._make_union_basis(tmp_circuit)
        empty_dag.gates = compiled_dag._make_union_gates(tmp_circuit)
        grouped_gates = group_gates(compiled_dag)
        # call mapper (based on an A* search) to satisfy the constraints for CNOTs
        # given by the coupling_map
        compiled_dag = a_star_mapper.a_star_mapper(
            copy.deepcopy(grouped_gates),
            JAGcoupling_list2dict(self.coupling_map),
            coupling.size(),
            copy.deepcopy(empty_dag)
        )
        grouped_gates_compiled = group_gates(compiled_dag)

        # estimate the cost of the mapped circuit:
        # the number of groups as well as the cost regarding to gate_costs
        min_groups = grouped_gates_compiled.order()
        min_cost = 0
        for operator, count in compiled_dag.count_ops().items():
            min_cost += count * self.gate_costs[operator]

        # Allow 30 seconds for this optimization or 1000 iterations (max)/9 more min.
        # The optimization is probabilistic, so a little more time can yield a better
        # solution, a shorter run-time, and a higher-fidelity result on real HW.
        stop = timer()
        elapsed = stop - start
        reps = int(30/elapsed)
        if reps < 9:
            reps = 9
        if reps > 1000:
            reps = 1000

        # Repeat the mapping procedure reps (>= 9) times, take the result with minimum cost.
        # Each call may yield a different result, since the mapper is implemented with a certain
        # non-determinism. In fact, in the priority queue used for implementing the A* algorithm,
        # the entries are a pair of the priority and a pointer to an object holding th mapping
        # infomation (as second criterion). Thus, it is uncertain which node is expanded first
        # if two nodes have the same priority (it depends on the value of the pointer). However,
        # this non-determinism allows to find different solution by repeatedly calling the mapper.
        for _ in range(reps):
            result = a_star_mapper.a_star_mapper(
                copy.deepcopy(grouped_gates),
                JAGcoupling_list2dict(self.coupling_map),
                coupling.size(),
                copy.deepcopy(empty_dag)
            )
            grouped_gates_result = group_gates(result)

            groups = grouped_gates_result.order()
            cost = 0
            for operator, count in result.count_ops().items():
                cost += count * self.gate_costs[operator]

            # take the solution with fewer groups (fewer cost if the number of groups is equal)
            if groups < min_groups or (groups == min_groups and cost < min_cost):
                min_groups = groups
                min_cost = cost
                compiled_dag = result
                grouped_gates_compiled = grouped_gates_result

        # post-mapping optimization: build 4x4 matrix for gate groups and decompose them
        # using KAK decomposition.  Moreover, subsequent single qubit gates are optimized.
        compiled_dag = post_mapping_optimization.optimize_gate_groups(
            grouped_gates_compiled,
            coupling.get_edges(),
            copy.deepcopy(empty_dag),
            self.gate_costs,
        )

        return compiled_dag
