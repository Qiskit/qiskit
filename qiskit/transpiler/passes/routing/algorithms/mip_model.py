"""Integer programming model for quantum circuit compilation."""

import logging
import math

import cplex
import numpy as np
from cplex import SparsePair

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.quantum_info import two_qubit_cnot_decompose
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition, \
    trace_to_fid
from qiskit.transpiler.layout import Layout

logger = logging.getLogger(__name__)


class MIPMappingModel:
    """Internal circuit and topology model to create a MIP problem for mapping
    """
    def __init__(self, dag, coupling_map, basis_fidelity, fixed_layout=False):
        initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        self.virtual_to_index = {v: i for i, v in enumerate(initial_layout.get_virtual_bits())}
        self.index_to_virtual = {i: v for i, v in enumerate(initial_layout.get_virtual_bits())}

        # 2-qubit gates
        self.gates = []
        # Fidelities of original gate
        gate_fidelity = []
        # Fidelities of the mirrored gate
        gate_mfidelity = []
        # Generate data structures for the optimization problem
        for t, lay in enumerate(dag.layers()):
            laygates = []
            layfidelity = []
            laymfidelity = []
            subdag = lay['graph']
            for node in subdag.two_qubit_ops():
                matrix = node.op.to_matrix()
                swap = SwapGate().to_matrix()
                if basis_fidelity:
                    target = TwoQubitWeylDecomposition(matrix)
                    targetm = TwoQubitWeylDecomposition(matrix @ swap)
                    traces = two_qubit_cnot_decompose.traces(target)
                    tracesm = two_qubit_cnot_decompose.traces(targetm)
                    fidelity = [trace_to_fid(traces[i]) for i in range(4)]
                    mfidelity = [trace_to_fid(tracesm[i]) for i in range(4)]
                else:
                    fidelity = [0.01, 0.01, 1.0, 0.01]
                    mfidelity = [0.01, 0.01, 0.01, 1.0]
                i1 = self.virtual_to_index[node.qargs[0]]
                i2 = self.virtual_to_index[node.qargs[1]]
                laygates.append((i1, i2))
                layfidelity.append(fidelity)
                laymfidelity.append(mfidelity)
            if laygates:
                self.gates.append(laygates)
                gate_fidelity.append(layfidelity)
                gate_mfidelity.append(laymfidelity)

        # If we use a fixed layout, we add an empty layer of gates to
        # allow our MILP compiler to add swaps; otherwise, the MILP may be infeasible
        if fixed_layout:
            self.gates = [[]] + self.gates
            gate_fidelity = [[]] + gate_fidelity
            gate_mfidelity = [[]] + gate_mfidelity

        self.circuit_model = _CircuitModel(
            num_qubits=len(dag.qubits),
            depth=len(self.gates),
            gates=self.gates,
            gate_fidelity=gate_fidelity,
            gate_mfidelity=gate_mfidelity)

        self.toplogy_model = _HardwareTopology(
            num_qubits=coupling_map.size(),
            connectivity=[list(coupling_map.neighbors(i))
                          for i in range(coupling_map.size())],
            basis_fidelity=[[basis_fidelity
                             for _ in coupling_map.neighbors(i)]
                            for i in range(coupling_map.size())])

    def create_cpx_problem(self, dummy_time_steps,
                           max_total_dummy=10000, line_symm=False, cycle_symm=False):
        """Create integer programming model to compile a circuit.

        Parameters
        ----------
        dummy_time_steps : int
            Number of dummy time steps, after each real layer of gates, to
            allow arbitrary swaps between neighbors.

        max_total_dummy : int
            Maximum allowed number of nonempty dummy time steps

        line_symm : bool
            Use symmetry breaking constrainst for line topology. Should
            only be True if the hardware graph is a chain/line/path.

        cycle_symm : bool
            Use symmetry breaking constrainst for loop. Should only be
            True if the hardware graph is a cycle/loop/ring.

        Returns
        -------
        cplex.Cplex, IndexCalculator
            A Cplex problem object with the problem description, and an
            index calculator to retrieve variable indices.

        """
        prob = cplex.Cplex()
        ic = IndexCalculator(self.circuit_model, self.toplogy_model,
                             dummy_time_steps)
        # Add w variables
        w_names = [None] * ic.num_w_vars
        for l in range(ic.num_lqubits):
            for p in range(ic.num_pqubits):
                for t in range(ic.depth):
                    w_names[ic.w_index(l, p, t)] = 'w' + \
                                                   '_{:d}_{:d}_{:d}'.format(l, p, t)
        prob.variables.add(
            types=[prob.variables.type.binary] * ic.num_w_vars,
            names=w_names)
        # Add y variables
        y_names = [None] * ic.num_y_vars
        for t in range(ic.depth):
            for (p, q) in ic.qc.gates[t]:
                for (i, j) in ic.ht.arcs:
                    y_names[ic.y_index(t, (p, q), (i, j)) -
                            ic.y_start] = 'y_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                        t, p, q, i, j)
        prob.variables.add(
            types=[prob.variables.type.binary] * ic.num_y_vars,
            names=y_names)
        # Add x variables
        x_names = [None] * ic.num_x_vars
        for q in range(ic.num_lqubits):
            for i in range(ic.num_pqubits):
                for j in ic.ht.outstar_by_node[i]:
                    for t in range(ic.depth - 1):
                        x_names[ic.x_index(q, i, j, t) -
                                ic.x_start] = 'x' + \
                                              '_{:d}_{:d}_{:d}_{:d}'.format(q, i, j, t)
        prob.variables.add(
            types=[prob.variables.type.binary] * ic.num_x_vars,
            names=x_names)
        # Add w^t variables (also denoted w^d here)
        prob.variables.add(
            types=[prob.variables.type.binary] * ic.num_wd_vars,
            names=['w_{:d}'.format(t) for t in range(ic.depth)])
        # Add e^u variables (edge used)
        eu_names = [None] * ic.num_eu_vars
        for t in range(ic.depth - 1):
            for (i, j) in ic.ht.edges:
                eu_names[ic.eu_index(t, (i, j)) - ic.eu_start] = 'eu_{:d}_{:d}_{:d}'.format(t, i, j)
        prob.variables.add(
            types=[prob.variables.type.binary] * ic.num_eu_vars,
            names=eu_names)
        # Assignment constraints for w variables
        prob.linear_constraints.add(
            lin_expr=[SparsePair(
                ind=[ic.w_index(q, j, t) for j in range(ic.num_pqubits)],
                val=[1 for j in range(ic.num_pqubits)])
                for q in range(ic.num_lqubits) for t in range(ic.depth)],
            senses=['E' for q in range(ic.num_lqubits) for t in range(ic.depth)],
            rhs=[1 for q in range(ic.num_lqubits) for t in range(ic.depth)],
            names=['assignment_lqubits_{:d}_{:d}'.format(q, t)
                   for q in range(ic.num_lqubits) for t in range(ic.depth)])
        prob.linear_constraints.add(
            lin_expr=[SparsePair(
                ind=[ic.w_index(q, j, t) for q in range(ic.num_lqubits)],
                val=[1 for q in range(ic.num_lqubits)])
                for j in range(ic.num_pqubits) for t in range(ic.depth)],
            senses=['E' for j in range(ic.num_pqubits) for t in range(ic.depth)],
            rhs=[1 for j in range(ic.num_pqubits) for t in range(ic.depth)],
            names=['assignment_pqubits_{:d}_{:d}'.format(j, t)
                   for j in range(ic.num_pqubits) for t in range(ic.depth)])
        # Link between w variables and y variables, i.e. a gate can only
        # be implemented on an arc that has the right logical qubits at
        # its endpoints
        for t in range(ic.depth):
            for (p, q) in ic.qc.gates[t]:
                for i in range(ic.num_pqubits):
                    for j in ic.ht.outstar_by_node[i]:
                        if (t == ic.depth - 1):
                            prob.linear_constraints.add(
                                lin_expr=[SparsePair(
                                    ind=[ic.w_index(p, i, t)] +
                                        [ic.y_index(t, (p, q), (i, j))],
                                    val=[-1, 1])],
                                senses=['L'], rhs=[0],
                                names=['w_y_p_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                                    t, p, q, i, j)])
                            prob.linear_constraints.add(
                                lin_expr=[SparsePair(
                                    ind=[ic.w_index(q, j, t)] +
                                        [ic.y_index(t, (p, q), (i, j))],
                                    val=[-1, 1])],
                                senses=['L'], rhs=[0],
                                names=['w_y_q_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                                    t, p, q, i, j)])
                        else:
                            prob.linear_constraints.add(
                                lin_expr=[SparsePair(
                                    ind=[ic.x_index(p, i, i, t),
                                         ic.x_index(p, i, j, t),
                                         ic.y_index(t, (p, q), (i, j))],
                                    val=[-1, -1, 1])],
                                senses=['L'], rhs=[0],
                                names=['w_y_p_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                                    t, p, q, i, j)])
                            prob.linear_constraints.add(
                                lin_expr=[SparsePair(
                                    ind=[ic.x_index(q, j, j, t),
                                         ic.x_index(q, j, i, t),
                                         ic.y_index(t, (p, q), (i, j))],
                                    val=[-1, -1, 1])],
                                senses=['L'], rhs=[0],
                                names=['w_y_q_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                                    t, p, q, i, j)])
        # Each gate must be implemented
        for t in range(ic.depth):
            for (p, q) in ic.qc.gates[t]:
                prob.linear_constraints.add(
                    lin_expr=[SparsePair(
                        ind=[ic.y_index(t, (p, q), a) for a in ic.ht.arcs],
                        val=[1 for a in ic.ht.arcs])],
                    senses=['E'], rhs=[1],
                    names=['assignment_y_{:d}_{:d}_{:d}'.format(t, p, q)])
        # If a gate is implemented, involved qubits cannot swap with other
        # positions
        for t in range(ic.depth - 1):
            for (p, q) in ic.qc.gates[t]:
                for (i, j) in ic.ht.arcs:
                    prob.linear_constraints.add(
                        lin_expr=[SparsePair(
                            ind=[ic.x_index(p, i, j, t),
                                 ic.x_index(q, j, i, t)],
                            val=[1, -1])],
                        senses=['E'], rhs=[0],
                        names=['swap_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
                            t, p, q, i, j)])
        # Qubit not in gates can flip with their neighbors
        for t in range(ic.depth - 1):
            q_no_gate = [i for i in range(ic.num_lqubits)]
            for (p, q) in ic.qc.gates[t]:
                q_no_gate.remove(p)
                q_no_gate.remove(q)
            for (i, j) in ic.ht.arcs:
                prob.linear_constraints.add(
                    lin_expr=[SparsePair(
                        ind=[ic.x_index(q, i, j, t) for q in q_no_gate] +
                            [ic.x_index(p, j, i, t) for p in q_no_gate],
                        val=[1 for q in q_no_gate] + [-1 for q in q_no_gate])],
                    senses=['E'], rhs=[0],
                    names=['swap_no_gate_{:d}_{:d}_{:d}'.format(t, i, j)])
        # Count non-dummy time steps
        non_dummy = 0
        # See if a dummy time step is needed
        for t in range(ic.depth):
            # This is a dummy time step
            if (len(ic.qc.gates[t]) == 0):
                prob.linear_constraints.add(
                    lin_expr=[SparsePair(
                        ind=[ic.x_index(p, i, j, t) for (i, j) in ic.ht.arcs] +
                            [ic.wd_index(t)],
                        val=[1 for (i, j) in ic.ht.arcs] + [-1])
                        for p in range(ic.num_lqubits)],
                    senses=['L' for p in range(ic.num_lqubits)],
                    rhs=[0 for p in range(ic.num_lqubits)],
                    names=['dummy_ts_needed_{:d}_{:d}'.format(t, p)
                           for p in range(ic.num_lqubits)])
            else:
                prob.variables.set_lower_bounds(ic.wd_index(t), 1)
                non_dummy += 1
        # Total number of dummy steps
        prob.linear_constraints.add(
            lin_expr=[SparsePair(
                ind=[ic.wd_index(t) for t in range(ic.depth)],
                val=[1 for t in range(ic.depth)])],
            senses=['L'], rhs=[max_total_dummy + non_dummy],
            names=['dummy_ts_total'])
        # Symmetry breaking between dummy time steps
        for t in range(ic.depth - 1):
            # This is a dummy time step and the next one is dummy too
            if (len(ic.qc.gates[t]) == 0 and
                    len(ic.qc.gates[t + 1]) == 0):
                # We cannot use the next time step unless this one is used too
                prob.linear_constraints.add(
                    lin_expr=[SparsePair(
                        ind=[ic.wd_index(t), ic.wd_index(t + 1)],
                        val=[1, -1])],
                    senses=['G'], rhs=[0],
                    names=['dummy_precedence_{:d}'.format(t)])
        # Symmetry breaking on the line -- only works on line topology!
        if (line_symm):
            for h in range(1, ic.num_lqubits):
                prob.linear_constraints.add(
                    lin_expr=[SparsePair(
                        ind=[ic.w_index(p, 0, 0) for p in range(h)] +
                            [ic.w_index(q, ic.num_pqubits - 1, 0)
                             for q in range(h, ic.num_lqubits)],
                        val=[1 for p in range(ic.num_lqubits)])],
                    senses=['G'], rhs=[1],
                    names=['sym_break_line_{:d}'.format(h)])
        # Symmetry breaking on the cycle -- only works on cycle topology!
        if (cycle_symm):
            prob.variables.set_lower_bounds(ic.w_index(0, 0, 0), 1)
            # Logical qubit flow constraints
        for t in range(ic.depth):
            for q in range(ic.num_lqubits):
                if (t < ic.depth - 1):
                    # Flow out; skip last time step
                    prob.linear_constraints.add(
                        lin_expr=[SparsePair(
                            ind=[ic.w_index(q, i, t)] + [ic.x_index(q, i, i, t)] +
                                [ic.x_index(q, i, j, t)
                                 for j in ic.ht.outstar_by_node[i]],
                            val=[1, -1] + [-1 for j in ic.ht.outstar_by_node[i]])
                            for i in range(ic.num_pqubits)],
                        senses=['E' for i in range(ic.num_pqubits)],
                        rhs=[0 for i in range(ic.num_pqubits)],
                        names=['flow_out_{:d}_{:d}_{:d}'.format(t, q, i)
                               for i in range(ic.num_pqubits)])
                if (t > 0):
                    # Flow in; skip first time step
                    prob.linear_constraints.add(
                        lin_expr=[SparsePair(
                            ind=[ic.w_index(q, j, t)] +
                                [ic.x_index(q, j, j, t - 1)] +
                                [ic.x_index(q, i, j, t - 1)
                                 for i in ic.ht.instar_by_node[j]],
                            val=[1, -1] + [-1 for i in ic.ht.outstar_by_node[j]])
                            for j in range(ic.num_pqubits)],
                        senses=['E' for j in range(ic.num_pqubits)],
                        rhs=[0 for j in range(ic.num_pqubits)],
                        names=['flow_in_{:d}_{:d}_{:d}'.format(t, q, j)
                               for j in range(ic.num_pqubits)])
        # Definition of e^u variables
        for t in range(ic.depth - 1):
            used_qubit = [False for i in range(ic.num_lqubits)]
            edge_representation = [[] for (i, j) in ic.ht.edges]
            for (p, q) in ic.qc.gates[t]:
                used_qubit[p] = True
                used_qubit[q] = True
                for (k, (i, j)) in enumerate(ic.ht.edges):
                    edge_representation[k].append(ic.y_index(t, (p, q), (i, j)))
                    edge_representation[k].append(ic.y_index(t, (p, q), (j, i)))
            for (i, j) in ic.ht.edges:
                for q in range(ic.num_lqubits):
                    index = ic.ht.edge_to_index((i, j))
                    if (not used_qubit[q] and ic.x_index(q, i, j, t) not in edge_representation[index]):
                        edge_representation[index].append(ic.x_index(q, i, j, t))
            for (k, (i, j)) in enumerate(ic.ht.edges):
                prob.linear_constraints.add(
                    lin_expr=[SparsePair(
                        ind=edge_representation[k] + [ic.eu_index(t, (i, j))],
                        val=[1 for h in edge_representation[k]] + [-1])],
                    senses=['E'], rhs=[0],
                    names=['eu_def_{:d}_{:d}_{:d}'.format(t, i, j)])

        return prob, ic


def set_error_rate_obj(prob, ic):
    """Set the minimum error rate objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Set objective
    for t in range(ic.depth):
        used_qubit = [False for i in range(ic.num_lqubits)]
        used_qubit_fidelity = [0] * ic.num_lqubits
        used_qubit_mfidelity = [0] * ic.num_lqubits
        for (p, q) in ic.qc.gates[t]:
            used_qubit[p] = True
            used_qubit[q] = True
            used_qubit_fidelity[p] = ic.qc.gate_fidelity(t, (p, q))
            used_qubit_fidelity[q] = ic.qc.gate_fidelity(t, (p, q))
            used_qubit_mfidelity[p] = ic.qc.gate_mfidelity(t, (p, q))
            used_qubit_mfidelity[q] = ic.qc.gate_mfidelity(t, (p, q))
            for (i, j) in ic.ht.arcs:
                # We pay the cost for gate implementation. If we are
                # mirroring, another objective function coefficient
                # (below) will ensure we pay the difference between
                # regular cost and mirrored cost.
                expected_fidelities = [used_qubit_fidelity[p][k] *
                                       ic.ht.arc_basis_fidelity((i, j)) ** k
                                       for k in range(4)]
                pbest_fid = -np.log(np.max(expected_fidelities))
                prob.objective.set_linear(
                    ic.y_index(t, (p, q), (i, j)), pbest_fid)
        if (t < ic.depth - 1):
            # x variables are only defined for depth up to depth-1
            for i in range(ic.num_pqubits):
                for q in range(ic.num_lqubits):
                    for j in ic.ht.outstar_by_node[i]:
                        if (not used_qubit[q]):
                            # This means we are swapping
                            prob.objective.set_linear(
                                ic.x_index(q, i, j, t),
                                -3 / 2 * ic.ht.arc_log_basis_fidelity((i, j)))
                        else:
                            # This is a mirrored gate, so compute the
                            # difference between its mirrored cost and the
                            # non-mirrored cost
                            expected_fidelities = [
                                used_qubit_fidelity[q][k] *
                                ic.ht.arc_basis_fidelity((i, j)) ** k
                                for k in range(4)]
                            expected_fidelitiesm = [
                                used_qubit_mfidelity[q][k] *
                                ic.ht.arc_basis_fidelity((i, j)) ** k
                                for k in range(4)]
                            pbest_fid = -np.log(np.max(expected_fidelities))
                            pbest_fidm = -np.log(np.max(expected_fidelitiesm))
                            prob.objective.set_linear(
                                ic.x_index(q, i, j, t),
                                (pbest_fidm - pbest_fid) / 2)


# -- end function

def unset_error_rate_obj(prob, ic):
    """Unset the error rate objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Reset objective
    for i in range(ic.x_start, ic.x_start + ic.num_x_vars):
        prob.objective.set_linear(i, 0)
    for i in range(ic.y_start, ic.y_start + ic.num_y_vars):
        prob.objective.set_linear(i, 0)
    # -- end function


def set_error_rate_constraint(prob, ic, rhs):
    """Set a maximum error rate constraint.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    rhs : float
        Maximum allowed error rate

    """
    # Create constraint
    indices = []
    values = []
    # Set objective
    for t in range(ic.depth):
        used_qubit = [False for i in range(ic.num_lqubits)]
        used_qubit_fidelity = [0] * ic.num_lqubits
        used_qubit_mfidelity = [0] * ic.num_lqubits
        for (p, q) in ic.qc.gates[t]:
            used_qubit[p] = True
            used_qubit[q] = True
            used_qubit_fidelity[p] = ic.qc.gate_fidelity(t, (p, q))
            used_qubit_fidelity[q] = ic.qc.gate_fidelity(t, (p, q))
            used_qubit_mfidelity[p] = ic.qc.gate_mfidelity(t, (p, q))
            used_qubit_mfidelity[q] = ic.qc.gate_mfidelity(t, (p, q))
            for (i, j) in ic.ht.arcs:
                # We pay the cost for gate implementation. If we are
                # mirroring, another objective function coefficient
                # (below) will ensure we pay the difference between
                # regular cost and mirrored cost.
                expected_fidelities = [used_qubit_fidelity[p][k] *
                                       ic.ht.arc_basis_fidelity((i, j)) ** k
                                       for k in range(4)]
                pbest_fid = -np.log(np.max(expected_fidelities))
                indices.append(ic.y_index(t, (p, q), (i, j)))
                values.append(pbest_fid)
        if (t < ic.depth - 1):
            for i in range(ic.num_pqubits):
                for q in range(ic.num_lqubits):
                    for j in ic.ht.outstar_by_node[i]:
                        if (not used_qubit[q]):
                            # This means we are swapping
                            indices.append(ic.x_index(q, i, j, t))
                            values.append(-3 / 2 *
                                          ic.ht.arc_log_basis_fidelity((i, j)))
                        else:
                            # This is a mirrored gate, so compute the
                            # difference between its mirrored cost and the
                            # non-mirrored cost
                            expected_fidelities = [
                                used_qubit_fidelity[q][k] *
                                ic.ht.arc_basis_fidelity((i, j)) ** k
                                for k in range(4)]
                            expected_fidelitiesm = [
                                used_qubit_mfidelity[q][k] *
                                ic.ht.arc_basis_fidelity((i, j)) ** k
                                for k in range(4)]
                            pbest_fid = -np.log(np.max(expected_fidelities))
                            pbest_fidm = -np.log(np.max(expected_fidelitiesm))
                            indices.append(ic.x_index(q, i, j, t))
                            values.append((pbest_fidm - pbest_fid) / 2)
    prob.linear_constraints.add(
        lin_expr=[SparsePair(ind=indices, val=values)],
        senses=['L'], rhs=[rhs], names=['error_rate_const'])


# -- end function

def unset_error_rate_constraint(prob, ic):
    """Unset the maximum error rate constraint.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Add constraint
    prob.linear_constraints.delete(['error_rate_const'])


# -- end function

def set_depth_obj(prob, ic):
    """Set the minimum depth objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Set objective
    for i in range(ic.wd_start, ic.wd_start + ic.num_wd_vars):
        prob.objective.set_linear(i, 1)


# -- end function

def unset_depth_obj(prob, ic):
    """Unset the depth objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Reset objective
    for i in range(ic.wd_start, ic.wd_start + ic.num_wd_vars):
        prob.objective.set_linear(i, 0)


# -- end function

def set_depth_constraint(prob, ic, rhs):
    """Set a maximum depth constraint.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    rhs : float
        Maximum depth

    """
    # Add constraint
    prob.linear_constraints.add(
        lin_expr=[SparsePair(
            ind=[ic.wd_index(t) for t in range(ic.depth)],
            val=[1 for t in range(ic.depth)])],
        senses=['L'], rhs=[rhs], names=['depth_const'])


# -- end function

def unset_depth_constraint(prob, ic):
    """Unset the maximum depth constraint.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Add constraint
    prob.linear_constraints.delete(['depth_const'])


# -- end function

def set_cross_talk_obj(prob, ic):
    """Set the minimum cross talk objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Set objective
    for t in range(ic.depth - 1):
        for e1 in ic.ht.edges:
            for e2 in ic.ht.cross_talk(e1):
                prob.objective.set_quadratic_coefficients(
                    ic.eu_index(t, e1), ic.eu_index(t, e2), 1.0)


# -- end function

def unset_cross_talk_obj(prob, ic):
    """Unset the minimum cross talk objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    prob.objective.set_quadratic([0] * prob.variables.get_num())


# -- end function

def set_cross_talk_constraint(prob, ic, rhs):
    """Set the maximum cross talk constraint.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    rhs : float
        Maximum cross-talk.

    """
    product_vars = 0
    product_vars_start = ic.eu_start + ic.num_eu_vars
    # New variable index
    product_index = dict()
    # Store constraint coefficients here
    ind1 = list()
    ind2 = list()
    val = list()
    for t in range(ic.depth - 1):
        for e1 in ic.ht.edges:
            for e2 in ic.ht.cross_talk(e1):
                if ((t, ic.eu_index(t, e1), ic.eu_index(t, e2)) not in product_index and
                        (t, ic.eu_index(t, e2), ic.eu_index(t, e1)) not in product_index):
                    product_index[(t, ic.eu_index(t, e1), ic.eu_index(t, e2))] = product_vars
                    product_vars += 1
                    prob.variables.add(
                        types=[prob.variables.type.binary],
                        names=['aux_{:d}_{:d}_{:d}'.format(
                            t, ic.eu_index(t, e1), ic.eu_index(t, e2))])
                    prob.linear_constraints.add(
                        lin_expr=[SparsePair(
                            ind=[ic.eu_index(t, e1), ic.eu_index(t, e2),
                                 product_vars_start + product_vars - 1],
                            val=[1, 1, -1])],
                        senses=['L'], rhs=[1],
                        names=['mccormick_0_{:d}_{:d}'.format(
                            ic.eu_index(t, e1), ic.eu_index(t, e2))])
                    prob.linear_constraints.add(
                        lin_expr=[SparsePair(
                            ind=[ic.eu_index(t, e1),
                                 product_vars_start + product_vars - 1],
                            val=[-1, 1])],
                        senses=['L'], rhs=[0],
                        names=['mccormick_1_{:d}_{:d}'.format(
                            ic.eu_index(t, e1), ic.eu_index(t, e2))])
                    prob.linear_constraints.add(
                        lin_expr=[SparsePair(
                            ind=[ic.eu_index(t, e2),
                                 product_vars_start + product_vars - 1],
                            val=[-1, 1])],
                        senses=['L'], rhs=[0],
                        names=['mccormick_1_{:d}_{:d}'.format(
                            ic.eu_index(t, e1), ic.eu_index(t, e2))])
    prob.linear_constraints.add(
        lin_expr=[SparsePair(
            ind=[product_vars_start + j for j in range(product_vars)],
            val=[1 for j in range(product_vars)])],
        senses=['L'], rhs=[rhs], names=['cross_talk'])


# -- end function

def unset_cross_talk_constraint(prob, ic):
    """Unset the maximum cross talk constraint.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the model

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    rhs : float
        Maximum cross-talk.

    """
    # Unset constraint
    prob.linear_constraints.delete('cross_talk')
    prob.linear_constraints.delete(
        [i for (i, c) in enumerate(prob.linear_constraints.get_names()) if
         c.startswith('mccormick')])
    prob.variables.delete([v for v in prob.variables.get_names() if v.startswith('aux')])


# -- end function

def evaluate_error_rate_obj(prob, ic):
    """Evaluate the minimum error rate objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the solution

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    sol = prob.solution.get_values()
    obj = 0
    # Loop over terms to ocmpute objective
    for t in range(ic.depth - 1):
        used_qubit = [False for i in range(ic.num_lqubits)]
        used_qubit_fidelity = [0] * ic.num_lqubits
        used_qubit_mfidelity = [0] * ic.num_lqubits
        for (p, q) in ic.qc.gates[t]:
            used_qubit[p] = True
            used_qubit[q] = True
            used_qubit_fidelity[p] = ic.qc.gate_fidelity(t, (p, q))
            used_qubit_fidelity[q] = ic.qc.gate_fidelity(t, (p, q))
            used_qubit_mfidelity[p] = ic.qc.gate_mfidelity(t, (p, q))
            used_qubit_mfidelity[q] = ic.qc.gate_mfidelity(t, (p, q))
            for (i, j) in ic.ht.arcs:
                # We pay the cost for gate implementation. If we are
                # mirroring, another objective function coefficient
                # (below) will ensure we pay the difference between
                # regular cost and mirrored cost.
                expected_fidelities = [used_qubit_fidelity[p][k] *
                                       ic.ht.arc_basis_fidelity((i, j)) ** k
                                       for k in range(4)]
                pbest_fid = -np.log(np.max(expected_fidelities))
                obj += (sol[ic.y_index(t, (p, q), (i, j))] *
                        pbest_fid)
        for i in range(ic.num_pqubits):
            for q in range(ic.num_lqubits):
                for j in ic.ht.outstar_by_node[i]:
                    if (not used_qubit[q]):
                        # This means we are swapping
                        obj += (sol[ic.x_index(q, i, j, t)] *
                                -3 / 2 * ic.ht.arc_log_basis_fidelity((i, j)))
                    else:
                        # This is a mirrored gate, so compute the
                        # difference between its mirrored cost and the
                        # non-mirrored cost
                        expected_fidelities = [
                            used_qubit_fidelity[q][k] *
                            ic.ht.arc_basis_fidelity((i, j)) ** k
                            for k in range(4)]
                        expected_fidelitiesm = [
                            used_qubit_mfidelity[q][k] *
                            ic.ht.arc_basis_fidelity((i, j)) ** k
                            for k in range(4)]
                        pbest_fid = -np.log(np.max(expected_fidelities))
                        pbest_fidm = -np.log(np.max(expected_fidelitiesm))
                        obj += (sol[ic.x_index(q, i, j, t)] *
                                (pbest_fidm - pbest_fid) / 2)
    return obj


# -- end function

def evaluate_depth_obj(prob, ic):
    """Evaluate the minimum depth objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the solution

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    # Compute objective
    obj = 0
    for i in range(ic.wd_start, ic.wd_start + ic.num_wd_vars):
        obj += prob.solution.get_values(i)
    return obj


# -- end function

def evaluate_cross_talk_obj(prob, ic):
    """Evaluate the cross talk objective function.

    Parameters
    ----------
    prob : `cplex.Cplex`
        A Cplex problem object containg the solution

    ic : `IndexCalculator`
        Corresponding index calculator to retrieve variable indices

    """
    obj = 0
    sol = prob.solution.get_values()
    # Compute objective
    for t in range(ic.depth - 1):
        for e1 in ic.ht.edges:
            for e2 in ic.ht.cross_talk(e1):
                # Note: we multiply by 0.5 because if e2 is in
                # cross_talk(e1), then e1 is in cross_talk(e2)
                obj += sol[ic.eu_index(t, e1)] * sol[ic.eu_index(t, e2)] * 0.5
    return obj


# -- end function


def solve_cpx_model(problem, index_calculator, time_limit=300,
                    heuristic_emphasis=False, silent=False):
    """Solve the Cplex model and print statistics.

    Parameters
    ----------
    problem : cplex.Cplex
        Model implemented in a Cplex object.

    index_calculator : IndexCalculator
        Index calculator used to construct the model.

    time_limit : float
        Time limit given to Cplex.

    heuristic_emphasis : bool
        Focus on getting good solutions rather than proving optimality

    silent : bool
        Disable output printing.
    """
    problem.parameters.timelimit.set(time_limit)
    problem.parameters.preprocessing.qtolin.set(1)
    # This sets the number of threads; by default Cplex uses everything!
    # problem.parameters.threads.set(8)
    if (heuristic_emphasis):
        problem.parameters.emphasis.mip.set(5)
    if (silent):
        problem.set_log_stream(None)
        problem.set_error_stream(None)
        problem.set_warning_stream(None)
        problem.set_results_stream(None)
    problem.solve()
    if (silent):
        return
    # Everything below is just for display purposes; could be
    # eliminated
    print()
    print('Time steps that can be eliminated:')
    for t in range(index_calculator.depth):
        if (problem.solution.get_values(index_calculator.wd_index(t)) > 0.5):
            print('  ', end='')
        else:
            print(' *', end='')
        if (t < index_calculator.depth - 1):
            print('->', end='')
    print()
    for q in range(index_calculator.true_num_lqubits):
        print('Position of qubit {:d}:'.format(q))
        for t in range(index_calculator.depth):
            pos = -1
            for i in range(index_calculator.num_pqubits):
                if (problem.solution.get_values(
                        index_calculator.w_index(q, i, t)) > 0.5):
                    pos = i
            print('{:>2d}'.format(pos), end='')
            if (t < index_calculator.depth - 1):
                print('->', end='')
            else:
                print()
    # Verify circuit
    for t in range(index_calculator.depth):
        for (p, q) in index_calculator.qc.gates[t]:
            for (i, j) in index_calculator.ht.arcs:
                if (problem.solution.get_values(
                        index_calculator.y_index(t, (p, q), (i, j))) > 0.5):
                    if (problem.solution.get_values(
                            index_calculator.w_index(p, i, t)) < 0.5 or
                            problem.solution.get_values(
                                index_calculator.w_index(q, j, t)) < 0.5):
                        print(
                            'Qubits {:d} and {:d} not in the right position at time {:d}'.format(p,
                                                                                                 q,
                                                                                                 t))
    num_swaps = 0
    num_merged_swaps = 0
    depth = 0
    for t in range(index_calculator.depth - 1):
        for (i, j) in index_calculator.ht.arcs:
            gate = False
            for (p, q) in index_calculator.qc.gates[t]:
                if (problem.solution.get_values(
                        index_calculator.y_index(t, (p, q), (i, j))) > 0.5
                        or
                        problem.solution.get_values(
                            index_calculator.y_index(t, (p, q), (j, i))) > 0.5):
                    gate = True
            for q in range(index_calculator.num_lqubits):
                if (problem.solution.get_values(
                        index_calculator.x_index(q, i, j, t)) > 0.5):
                    if (gate):
                        num_merged_swaps += 1
                    else:
                        num_swaps += 1
        if (problem.solution.get_values(index_calculator.wd_index(t)) > 0.5):
            depth += 1
    print('Num swaps:', num_swaps // 2, 'num merged swaps:', num_merged_swaps // 2, 'total depth:',
          depth)


class _CircuitModel:
    """Description of a quantum circuit.

    Parameters
    ----------

    num_qubits : int
        Number of qubits in the circuit.

    depth : int
        Depth of the circuit.

    gates : list[list[(int, int)]]
        A list of length equal to depth. Each element is a list of
        gates to be applied at that depth. Gates should be
        non-overlapping in terms of qubits. A gate is specified by two
        integers between 0 and num_qubits-1.

    gate_fidelity : list[list[int]]
        A list of the same dimensions as `gates`, indicating the
        fidelity of each gate when using 0, 1, 2 or 3 entangling
        gates. If it is not provided, it is assumed that the fidelity
        is 1 for 2 gates and 0.01 otherwise.

    gate_mfidelity : list[list[int]]
        A list of the same dimensions as `gates`, indicating the
        fidelity of each mirrored gate when using 0, 1, 2 or 3 entangling
        gates. If it is not provided, it is assumed that the fidelity
        is 1 for 3 gates and 0.01 otherwise.

    """

    def __init__(self, num_qubits, depth, gates, gate_fidelity=None,
                 gate_mfidelity=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.gates = gates
        self._orig_gate_fidelity = gate_fidelity
        self._orig_gate_mfidelity = gate_mfidelity
        self._gate_to_index = {(0, gate[0], gate[1]) : i
                               for i, gate in enumerate(gates[0])}
        for t in range(1, depth):
            for gate in gates[t]:
                self._gate_to_index[(t, gate[0], gate[1])] = len(self._gate_to_index)
        if (gate_fidelity is None or gate_mfidelity is None):
            self._gate_fidelity = {key : [0.01, 0.01, 1.0, 0.01] for key in self._gate_to_index.keys()}
            self._gate_mfidelity = {key : [0.01, 0.01, 1.0, 0.01] for key in self._gate_to_index.keys()}
        else:
            self._gate_fidelity = {(0, gate[0], gate[1]) : gate_fidelity[0][i]
                                  for (i, gate) in enumerate(gates[0])}
            self._gate_mfidelity = {(0, gate[0], gate[1]) : gate_mfidelity[0][i]
                                   for (i, gate) in enumerate(gates[0])}
            for t in range(1, depth):
                for (i, gate) in enumerate(gates[t]):
                    self._gate_fidelity[(t, gate[0], gate[1])] = gate_fidelity[t][i]
                    self._gate_mfidelity[(t, gate[0], gate[1])] = gate_mfidelity[t][i]

    def gate_to_index(self, t, gate):
        return self._gate_to_index[t, gate[0], gate[1]]

    def gate_fidelity(self, t, gate):
        return self._gate_fidelity[t, gate[0], gate[1]]

    def gate_mfidelity(self, t, gate):
        return self._gate_mfidelity[t, gate[0], gate[1]]


class _HardwareTopology:
    """Topology of a specific hardware.

    Parameters
    ----------

    num_qubits : int
        Number of qubits in the device

    connectivity : list[list[int]]
        A list of length num_qubits, containing for each qubit the
        other qubits to which it is connected. These edges are assumed
        to be directed.

    basis_fidelity : list[list[float]]
        Same size as `connectivity`, should contain a basis fidelity
        (success rate, between 0 and 1) for each two-qubit
        connection. If None, we assume the fidelity is 0.99 for each
        edge.

    cross_talk : list[(int, int, int, int)]
        Cross-talk information, where each entry is two pairs of
        (connected) qubits that cross-talk.
    """

    def __init__(self, num_qubits, connectivity, basis_fidelity=None,
                 cross_talk=None):
        self.num_qubits = num_qubits
        self.connectivity = connectivity
        self._orig_basis_fidelity = basis_fidelity
        edge_set = set()
        for u in range(num_qubits):
            for v in connectivity[u]:
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    edge_set.add((u, v))
        self.num_edges = len(edge_set)
        self.num_arcs = 2 * self.num_edges
        self.edges = [e for e in edge_set]
        self.arcs = ([(u, v) for (u, v) in edge_set] +
                     [(v, u) for (u, v) in edge_set])
        self.edges_by_node = [[k for k, (u, v) in enumerate(self.edges)
                               if (u == i or v == i)]
                              for i in range(self.num_qubits)]
        self.arcs_out_by_node = [[k for k, (u, v) in enumerate(self.arcs)
                                  if (u == i)]
                                 for i in range(self.num_qubits)]
        self.arcs_in_by_node = [[k for k, (u, v) in enumerate(self.arcs)
                                 if (v == i)]
                                for i in range(self.num_qubits)]
        self.outstar_by_node = [[v for k, (u, v) in enumerate(self.arcs)
                                 if (u == i)]
                                for i in range(self.num_qubits)]
        self.instar_by_node = [[u for k, (u, v) in enumerate(self.arcs)
                                if (v == i)]
                               for i in range(self.num_qubits)]
        self._edge_to_index = {edge: i for i, edge in enumerate(self.edges)}
        self._arc_to_index = dict()
        self._arc_to_index = {arc: i for i, arc in enumerate(self.arcs)}
        if (basis_fidelity is None):
            self._basis_fidelity = [[0.99] for i in range(num_qubits)
                                    for u in connectivity[i]]
        else:
            self._basis_fidelity = basis_fidelity
        self._edge_basis_fidelity = {(u, v): self._basis_fidelity[u][i]
                                     for u in range(num_qubits)
                                     for (i, v) in enumerate(connectivity[u])}
        self._arc_basis_fidelity = {(u, v): self._basis_fidelity[u][i]
                                    for u in range(num_qubits)
                                    for (i, v) in enumerate(connectivity[u])}
        self._arc_basis_fidelity.update({(v, u): self._basis_fidelity[u][i]
                                         for u in range(num_qubits)
                                         for (i, v) in enumerate(connectivity[u])})
        self._orig_cross_talk = None
        self._cross_talk = dict()
        if (cross_talk is not None):
            for i, j, u, v in cross_talk:
                # Ensure all edges are taken in the order indicated
                # stored in the edge data structure
                if (i, j) in self.edges:
                    a = (i, j)
                else:
                    a = (j, i)
                if (u, v) in self.edges:
                    b = (u, v)
                else:
                    b = (v, u)
                if (a not in self._cross_talk):
                    self._cross_talk[a] = [b]
                else:
                    self._cross_talk[a].append(b)
                if (b not in self._cross_talk):
                    self._cross_talk[b] = [a]
                else:
                    self._cross_talk[b].append(a)

    def edge_to_index(self, edge):
        return self._edge_to_index[edge]

    def arc_to_index(self, arc):
        return self._arc_to_index[arc]

    def arc_basis_fidelity(self, arc):
        return self._arc_basis_fidelity[arc]

    def edge_basis_fidelity(self, edge):
        return self._edge_basis_fidelity[edge]

    def arc_log_basis_fidelity(self, arc):
        return math.log(self._arc_basis_fidelity[arc])

    def edge_log_basis_fidelity(self, edge):
        return math.log(self._edge_basis_fidelity[edge])

    def cross_talk(self, edge):
        return (self._cross_talk[edge] if (edge) in self._cross_talk else [])


class IndexCalculator:
    """Compute flattened indices for decision variables.

    Parameters
    ----------
    quantum_circuit : `data_structures.QuantumCircuit`
        Description of a quantum circuit.

    hardware_topology : `data_structures.HardwareTopology`
        Description of a hardware topology.

    dummy_time_steps : int
        Number of dummy time steps, after each real layer of gates, to
        allow arbitrary swaps between neighbors.
    """
    def __init__(self,
                 quantum_circuit: _CircuitModel,
                 hardware_topology: _HardwareTopology,
                 dummy_time_steps):
        self.qc = self._add_dummy_time_steps(quantum_circuit, dummy_time_steps)
        logger.info(self.qc.num_qubits)
        logger.info(self.qc.depth)
        logger.info(self.qc.gates)
        self.ht = hardware_topology
        # True number of logical qubits
        self.true_num_lqubits = self.qc.num_qubits
        # Number of logical qubits is equal to the number of physical
        # qubits; some qubits could be dummy
        self.num_lqubits = self.ht.num_qubits
        # Number of physical qubits
        self.num_pqubits = self.ht.num_qubits
        self.depth = self.qc.depth
        self.num_gates = sum(len(self.qc.gates[t])
                             for t in range(self.qc.depth))
        logger.info('Num gates:', self.num_gates)
        self.num_arcs = len(self.ht.arcs)
        logger.info('Num arcs:', self.num_arcs)
        self.w_start = 0
        self.num_w_vars = self.depth*self.num_lqubits*self.num_pqubits
        self.y_start = self.w_start + self.num_w_vars
        self.num_y_vars = self.num_gates*self.num_arcs
        self.x_start = self.y_start + self.num_y_vars
        self.num_x_vars = ((self.depth-1)*self.num_lqubits*
                           (self.ht.num_arcs + self.num_pqubits))
        # Createmapping for x (flow) variables
        var_index = 0
        self.x_var_mapping = [dict() for i in range(self.num_pqubits)]
        for i in range(self.num_pqubits):
            self.x_var_mapping[i][i] = var_index
            var_index += 1
            for j in self.ht.outstar_by_node[i]:
                self.x_var_mapping[i][j] = var_index
                var_index += 1
        self.wd_start = self.x_start + self.num_x_vars
        self.num_wd_vars = self.depth
        self.eu_start = self.wd_start + self.num_wd_vars
        self.num_eu_vars = self.ht.num_edges * (self.depth-1)            
    # -- end function

    def w_index(self, logical_qubit, physical_qubit, gate_depth):
        """Return the index of a w variable.

        Parameters
        ----------
        logical_qubit : int
            Index of the logical qubit.

        physical_qubit : int
            Index of the physical qubit.

        gate_depth : int
            Depth of the gate.

        Returns
        -------
        int
            Index of the variable in a flattened (0-indexed) model.
        """
        return (self.w_start + gate_depth*self.num_lqubits*self.num_pqubits +
                physical_qubit*self.num_lqubits + logical_qubit)
    # -- end function
    
    def y_index(self, depth, gate, arc):
        """Return the index of a y variable.

        Parameters
        ----------
        depth : int
            Depth at which the gate is located.

        gate : (int, int)
            Gate, given as a pair of logical qubits.

        arc : (int, int)
            Arc, given as a pair of physical qubits.

        Returns
        -------
        int
            Index of the variable in a flattened (0-indexed) model.
        """

        gate_index = self.qc.gate_to_index(depth, gate)
        arc_index = self.ht.arc_to_index(arc)
        return (self.y_start + arc_index*self.num_gates + gate_index)
    # -- end function
    
    def x_index(self, logical_qubit, physical_qubit1, physical_qubit2,
                depth):
        """Return the index of an x variable.

        Parameters
        ----------
        logical_qubit : int
            Index of the logical qubit.

        physical_qubit1 : int
            Index of the physical qubit at which the logical qubit is
            located at depth depth.

        physical_qubit2 : int
            Index of the physical qubit at which the logical qubit is
            located at depth depth+1.

        depth : int
            Depth of the flow.

        Returns
        -------
        int
            Index of the variable in a flattened (0-indexed) model.
        """
        return (self.x_start +
                depth*self.num_lqubits*(self.ht.num_arcs +
                                             self.num_pqubits) +
                logical_qubit*(self.ht.num_arcs + self.num_pqubits) +
                self.x_var_mapping[physical_qubit1][physical_qubit2])
    # -- end function

    def wd_index(self, depth):
        """Return the index of the w^d variable.

        Parameters
        ----------
        depth : int
            Depth of the time step.

        Returns
        -------
        int
            Index of the variable in a flattened (0-indexed) model.
        """
        return self.wd_start + depth
    # -- end function

    def eu_index(self, depth, edge):
        """Return the index of the e^u variable.

        Parameters
        ----------
        depth : int
            Depth of the time step.

        edge : (int, int)
            Edge, given as a pair of physical qubits.

        Returns
        -------
        int
            Index of the variable in a flattened (0-indexed) model.
        """
        return self.eu_start + depth*(self.ht.num_edges) + self.ht.edge_to_index(edge)

    @staticmethod
    def _add_dummy_time_steps(quantum_circuit, dummy_time_steps):
        """Add dummy time steps to a circuit.

        Dummy time steps can only contain SWAPs.

        Parameters
        ----------
        quantum_circuit : `data_structures.QuantumCircuit`
            Description of a quantum circuit.

        dummy_time_steps : int
            Number of dummy time steps, after each real layer of gates, to
            allow arbitrary swaps between neighbors.

        Returns
        -------
        `data_structures.QuantumCircuit`
            Quantum circuit with dummy_time_steps between each layer.
        """
        num_qubits = quantum_circuit.num_qubits
        depth = quantum_circuit.depth
        gates = quantum_circuit.gates
        gate_fidelity = quantum_circuit._orig_gate_fidelity
        gate_mfidelity = quantum_circuit._orig_gate_mfidelity
        new_depth = 1 + (depth-1) * (1 + dummy_time_steps)
        new_gates = list()
        new_gate_fidelity = list()
        new_gate_mfidelity = list()
        for t in range(new_depth):
            if (t % (dummy_time_steps + 1) == 0):
                # This is a real time step: copy the layer
                old_t = t // (dummy_time_steps+1)
                new_gates.append(
                    [gate for gate in gates[old_t]])
                new_gate_fidelity.append(
                    [fid for fid in gate_fidelity[old_t]])
                new_gate_mfidelity.append(
                    [mfid for mfid in gate_mfidelity[old_t]])
            else:
                new_gates.append(list())
                new_gate_fidelity.append(list())
                new_gate_mfidelity.append(list())

        return _CircuitModel(num_qubits, new_depth, new_gates, new_gate_fidelity, new_gate_mfidelity)


