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

"""Map a DAGCircuit onto a `coupling_map` allocating qubits and adding swap gates."""
import copy
from typing import Optional

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGNode
from qiskit.quantum_info import two_qubit_cnot_decompose
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition, \
    trace_to_fid
from qiskit.transpiler import TransformationPass, CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.routing.algorithms import mip_model as qomp


class MIPMapping(TransformationPass):
    """Map a DAGCircuit onto a `coupling_map` allocating qubits and adding swap gates.

    The MIP mapper try to find the best layout and routing at the same time
    by solving a mathematical optimization problem represented as a MIP (Mixed Integer Programming).
    """

    def __init__(self,
                 coupling_map: CouplingMap,
                 silent: bool = False,
                 basis_fidelity: float = 0.96,
                 time_limit_er: float = 300,
                 time_limit_depth: float = 60,
                 dummy_steps: Optional[int] = None,
                 max_total_dummy: Optional[int] = None,
                 initial_layout=None,
                 heuristic_emphasis: bool = False,
                 line_symm : bool = False,
                 cycle_symm : bool = False):
        """MIPMapping initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
        """
        super().__init__()
        self.coupling_map = copy.deepcopy(coupling_map)  # save a copy since some methods modify it
        self.initial_layout = initial_layout
        self.dummy_steps = dummy_steps
        self.max_total_dummy = max_total_dummy
        # The failure rate is taken from QV64 paper
        self.topo = qomp._HardwareTopology(
            coupling_map.size(),
            [list(coupling_map.neighbors(i))
             for i in range(coupling_map.size())],
            [[basis_fidelity
              for _ in coupling_map.neighbors(i)]
             for i in range(coupling_map.size())])
        self.silent = silent
        self.basis_fidelity = basis_fidelity
        self.time_limit_er = time_limit_er
        self.time_limit_depth = time_limit_depth
        self.heuristic_emphasis = heuristic_emphasis
        self.line_symm = line_symm
        self.cycle_symm = cycle_symm

    def run(self, dag):
        """Run the MIPMapping pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if ...
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('MILP swapper runs on physical circuits.')

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError('More virtual qubits exist than physical.')

        # set parameters depending on initial layout or circuit
        self.initial_layout = self.initial_layout or self.property_set['layout']
        self.dummy_steps = self.dummy_steps or dag.num_qubits()
        self.max_total_dummy = self.max_total_dummy or (dag.num_qubits() * (dag.num_qubits() - 1))

        mapped_dag = self._copy_circuit_metadata(dag)

        gates = []
        # Fidelities of original gate
        gate_fidelity = []
        # Fidelities of the mirrored gate
        gate_mfidelity = []
        gateslookup = {}
        meas = []
        for t, lay in enumerate(dag.layers()):
            laygates = []
            layfidelity = []
            laymfidelity = []
            for node in lay['graph'].nodes():
                if len(node.qargs) == 2:
                    matrix = node.op.to_matrix()
                    swap = SwapGate().to_matrix()
                    if self.basis_fidelity:
                        target = TwoQubitWeylDecomposition(matrix)
                        targetm = TwoQubitWeylDecomposition(matrix @ swap)
                        traces = two_qubit_cnot_decompose.traces(target)
                        tracesm = two_qubit_cnot_decompose.traces(targetm)
                        fidelity = [trace_to_fid(traces[i]) for i in range(4)]
                        mfidelity = [trace_to_fid(traces[i]) for i in range(4)]
                    else:
                        fidelity = [0.01, 0.01, 1.0, 0.01]
                        mfidelity = [0.01, 0.01, 0.01, 1.0]
                    i1, i2 = node.qargs[0].index, node.qargs[1].index
                    laygates.append((i1, i2))
                    layfidelity.append(fidelity)
                    laymfidelity.append(mfidelity)
                    gateslookup[(t, i1, i2)] = node
                elif node.type == 'op':
                    meas.append(node)
            if laygates:
                gates.append(laygates)
                gate_fidelity.append(layfidelity)
                gate_mfidelity.append(laymfidelity)
        # If we use a fixed layout, we add an empty layer of gates to
        # allow our MILP compiler to add swaps; otherwise, the MILP
        # may be infeasible
        if (self.initial_layout is not None):
            gates = [[]] + gates
            gate_fidelity = [[]] + gate_fidelity
            gate_mfidelity = [[]] + gate_mfidelity
        circ = qomp._CircuitModel(len(dag.qubits), len(gates), gates, gate_fidelity, gate_mfidelity)
        problem, ic = qomp.create_cpx_model(circ, self.topo, self.dummy_steps,
                                            self.max_total_dummy, self.line_symm, self.cycle_symm)
        if (self.initial_layout is not None):
            layout = [0 for i in range(ic.num_lqubits)]
            for logical in self.initial_layout._v2p.keys():
                layout[logical.index] = self.initial_layout._v2p[logical]

            # Fix initial layout variables
            for i in range(ic.num_lqubits):
                problem.variables.set_lower_bounds(ic.w_index(i, layout[i], 0), 1)
            # Remove symmetry breaking, since the initial layout may
            # not satisfy it
            for h in range(1, ic.num_lqubits):
                problem.linear_constraints.delete(['sym_break_line_{:d}'.format(h)])

        qomp.set_error_rate_obj(problem, ic)

        qomp.solve_cpx_model(problem, ic, time_limit=self.time_limit_er,
                             heuristic_emphasis=self.heuristic_emphasis, silent=self.silent)
        if (not self.silent):
            print('OBJ1 Error rate:', qomp.evaluate_error_rate_obj(problem, ic),
                  'depth:', qomp.evaluate_depth_obj(problem, ic),
                  'cross-talk:', qomp.evaluate_cross_talk_obj(problem, ic))
        qomp.set_error_rate_constraint(problem, ic,
                                       problem.solution.get_objective_value())
        qomp.unset_error_rate_obj(problem, ic)

        qomp.set_depth_obj(problem, ic)
        qomp.solve_cpx_model(problem, ic, time_limit=self.time_limit_depth,
                             heuristic_emphasis=self.heuristic_emphasis, silent=self.silent)
        if (not self.silent):
            print('OBJ2 Error rate:', qomp.evaluate_error_rate_obj(problem, ic),
                  'depth:', qomp.evaluate_depth_obj(problem, ic),
                  'cross-talk:', qomp.evaluate_cross_talk_obj(problem, ic))
        # qomp.set_depth_constraint(problem, ic,
        #                           problem.solution.get_objective_value())
        # qomp.unset_depth_obj(problem, ic)

        for t in range(ic.depth):
            # If we use a fixed layout, the first layer of gates
            # is empty to allow swaps, therefore we shift all
            # indices by 1
            if (self.initial_layout is not None):
                lookup = t - self.dummy_steps + 1
            else:
                lookup = t
            for (i, j) in ic.ht.arcs:
                for (p, q) in ic.qc.gates[t]:
                    if problem.solution.get_values(ic.y_index(t, (p, q), (i, j))) > 0.5:
                        mapped_gate = self._transform_gate(
                            gateslookup[(lookup // (self.dummy_steps + 1), p, q)], i, j)
                        mapped_dag.apply_operation_back(mapped_gate.op, mapped_gate.qargs)

                if t < ic.depth - 1:
                    for q in range(ic.num_lqubits):
                        if (problem.solution.get_values(ic.x_index(q, i, j, t)) > 0.5) and i > j:
                            swap_node = DAGNode(op=SwapGate(), qargs=[dag.qubits[i], dag.qubits[j]],
                                                type='op')
                            mapped_dag.apply_operation_back(swap_node.op, swap_node.qargs)
        inmap = {}
        outmap = {}
        outmeas = [-1 for i in range(len(dag.qubits))]
        for l in range(len(dag.qubits)):
            inpos = None
            outpos = None
            for p in range(ic.num_pqubits):
                if problem.solution.get_values(ic.w_index(l, p, 0)) > 0.5:
                    inpos = p
                if problem.solution.get_values(ic.w_index(l, p, ic.depth - 1)) > 0.5:
                    outpos = p
                    outmeas[l] = p
            inmap[dag.qubits[l]] = inpos
            outmap[dag.qubits[l]] = outpos
        self.property_set['layout'] = Layout(inmap)
        self.property_set['layout_in'] = Layout(inmap)
        self.property_set['final_layout'] = Layout(outmap)
        self.property_set['layout_out'] = Layout(outmap)

        for m in meas:
            mapped_meas = copy.deepcopy(m)
            m.qargs = [m.qargs[0].register[outmeas[m.qargs[0].index]]]
            mapped_dag.apply_operation_back(op=mapped_meas.op, cargs=m.cargs, qargs=m.qargs)

        return mapped_dag

    @staticmethod
    def _copy_circuit_metadata(source_dag):
        """Return a copy of source_dag with metadata but empty.
        """
        target_dag = DAGCircuit()
        target_dag.name = source_dag.name

        for qreg in source_dag.qregs.values():
            target_dag.add_qreg(qreg)
        for creg in source_dag.cregs.values():
            target_dag.add_creg(creg)

        return target_dag

    @staticmethod
    def _transform_gate(op_node, i, j):
        """Return node implementing a virtual op on given layout."""
        mapped_op_node = copy.deepcopy(op_node)
        device_qreg = op_node.qargs[0].register
        mapped_op_node.qargs = [device_qreg[i], device_qreg[j]]

        return mapped_op_node
