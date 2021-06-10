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

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGNode
from qiskit.transpiler import TransformationPass, CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.routing.algorithms import mip_model as qomp
from qiskit.transpiler.passes.routing.algorithms.mip_model import MIPMappingModel


class MIPMapping(TransformationPass):
    """Map a DAGCircuit onto a `coupling_map` allocating qubits and adding swap gates.

    The MIP mapper try to find the best layout and routing at the same time
    by solving a mathematical optimization problem represented as a MIP (Mixed Integer Programming).
    """

    def __init__(self,
                 coupling_map: CouplingMap,
                 silent: bool = True,
                 basis_fidelity: float = 0.96,
                 time_limit_er: float = 30,
                 time_limit_depth: float = 30,
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
        # TODO: should run on logical circuits
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('MILP swapper runs on physical circuits only.')

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError('More virtual qubits exist than physical qubits.')

        canonical_register = dag.qregs['q']
        mapped_dag = dag._copy_circuit_metadata()

        # set parameters depending on initial layout or circuit
        self.initial_layout = self.initial_layout or self.property_set['layout']
        self.dummy_steps = self.dummy_steps or min(4, dag.num_qubits())
        self.max_total_dummy = self.max_total_dummy or (self.dummy_steps * (self.dummy_steps - 1))

        model = MIPMappingModel(dag=dag,
                                coupling_map=self.coupling_map,
                                initial_layout=self.initial_layout,
                                basis_fidelity=self.basis_fidelity)

        problem, ic = model.create_cpx_problem(
            self.dummy_steps,
            self.max_total_dummy,
            (self.line_symm and self.initial_layout is None),
            self.cycle_symm)

        if not model.do_layout:
            # Fix initial layout variables
            for q in range(ic.num_lqubits):
                i = self.initial_layout[model.index_to_virtual[q]]
                problem.variables.set_lower_bounds(ic.w_index(q, i, 0), 1)

        qomp.set_error_rate_obj(problem, ic)

        qomp.solve_cpx_model(problem, ic, time_limit=self.time_limit_er,
                             heuristic_emphasis=self.heuristic_emphasis, silent=self.silent)
        if not self.silent:
            print('OBJ1 Error rate:', qomp.evaluate_error_rate_obj(problem, ic),
                  'depth:', qomp.evaluate_depth_obj(problem, ic),
                  'cross-talk:', qomp.evaluate_cross_talk_obj(problem, ic))
        qomp.set_error_rate_constraint(problem, ic,
                                       problem.solution.get_objective_value())
        qomp.unset_error_rate_obj(problem, ic)

        qomp.set_depth_obj(problem, ic)
        qomp.solve_cpx_model(problem, ic, time_limit=self.time_limit_depth,
                             heuristic_emphasis=self.heuristic_emphasis, silent=self.silent)
        if not self.silent:
            print('OBJ2 Error rate:', qomp.evaluate_error_rate_obj(problem, ic),
                  'depth:', qomp.evaluate_depth_obj(problem, ic),
                  'cross-talk:', qomp.evaluate_cross_talk_obj(problem, ic))

        # Construct the circuit that includes routing
        for t in range(ic.depth):
            # If we use a fixed layout, the first layer of gates
            # is empty to allow swaps, therefore we shift all
            # indices by 1
            if not model.do_layout:
                lookup = t - self.dummy_steps + 1
            else:
                lookup = t
            for (i, j) in ic.ht.arcs:
                for (p, q) in ic.qc.gates[t]:
                    if problem.solution.get_values(ic.y_index(t, (p, q), (i, j))) > 0.5:
                        org_node = model.nodeslookup[(model.circuit_to_orig_layer[lookup // (self.dummy_steps + 1)], p, q)]
                        mapped_dag.apply_operation_back(
                            op=org_node.op,
                            qargs=[canonical_register[i], canonical_register[j]]
                        )
            for (i, j) in ic.ht.arcs:  # this loop ensure swaps come after 2q-gates for each layer
                if t < ic.depth - 1:
                    for q in range(ic.num_lqubits):
                        if (problem.solution.get_values(ic.x_index(q, i, j, t)) > 0.5) and i < j:
                            mapped_dag.apply_operation_back(
                                op=SwapGate(),
                                qargs=[canonical_register[i], canonical_register[j]]
                            )
        # Reconstruct the qubit map after the final layer: the qubits
        # could be permuted compared to the initial permutation.
        inmap = {}
        outmap = {}
        outmeas = [-1] * len(dag.qubits)
        for l, v in model.index_to_virtual.items():
            inpos = None
            outpos = None
            for p in range(ic.num_pqubits):
                if problem.solution.get_values(ic.w_index(l, p, 0)) > 0.5:
                    inpos = p
                if problem.solution.get_values(ic.w_index(l, p, ic.depth - 1)) > 0.5:
                    outpos = p
                    outmeas[l] = p
            inmap[v] = inpos
            outmap[v] = outpos
        # These are saved, but not sure if used anywhere
        self.property_set['layout'] = Layout(inmap)
        self.property_set['layout_in'] = Layout(inmap)
        self.property_set['final_layout'] = Layout(outmap)
        self.property_set['layout_out'] = Layout(outmap)
        # print(self.property_set)

        # Remap measurements
        for m in model.meas:
            mapped_meas = copy.deepcopy(m)
            l = model.virtual_to_index[m.qargs[0]]
            m.qargs = [canonical_register[outmeas[l]]]
            mapped_dag.apply_operation_back(op=mapped_meas.op, cargs=m.cargs, qargs=m.qargs)

        return mapped_dag

    @staticmethod
    def _transform_gate(op_node, i, j):
        """Return node implementing a virtual op on given layout."""
        mapped_op_node = copy.deepcopy(op_node)
        device_qreg = op_node.qargs[0].register
        mapped_op_node.qargs = [device_qreg[i], device_qreg[j]]

        return mapped_op_node
