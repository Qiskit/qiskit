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
        self.fixed_layout = False

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

        # canonical_register = dag.qregs['q']
        # mapped_dag = dag._create_empty_dagcircuit()
        canonical_register = QuantumRegister(self.coupling_map.size(), 'q')
        mapped_dag = self._create_empty_dagcircuit(dag, canonical_register)

        # set parameters depending on initial layout or circuit
        self.initial_layout = self.initial_layout or self.property_set['layout']
        if self.initial_layout is not None:
            self.fixed_layout = True
        self.dummy_steps = self.dummy_steps or min(4, dag.num_qubits())
        self.max_total_dummy = self.max_total_dummy or (self.dummy_steps * (self.dummy_steps - 1))

        model = MIPMappingModel(dag=dag,
                                coupling_map=self.coupling_map,
                                basis_fidelity=self.basis_fidelity,
                                fixed_layout=self.fixed_layout)

        problem, ic = model.create_cpx_problem(
            self.dummy_steps,
            self.max_total_dummy,
            (self.line_symm and self.initial_layout is None),
            self.cycle_symm)

        if self.fixed_layout:
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

        # create a layout to track changes in layout for each layer
        dic = {}
        for q in range(ic.num_lqubits):
            for i in range(ic.num_pqubits):
                if problem.solution.get_values(ic.w_index(q, i, 0)) > 0.5:
                    dic[model.index_to_virtual[q]] = i
        layout = Layout(dic)
        self.property_set['layout'] = copy.deepcopy(layout)
        # print("solution:", layout)

        # Construct the circuit that includes routing
        interval = self.dummy_steps + 1
        t_shift = 0  # TODO: Ideally the shift should be cared within MIPMappingModel
        if self.fixed_layout:
            t_shift = self.dummy_steps
        for k, layer in enumerate(dag.layers()):
            # add swaps between (k-1)-th and k-th the layer
            # dummy steps before 0-th layer are considered only in the fixed_layout case
            from_dummy_steps = 1 + interval * (k-1) + t_shift
            to_dummy_steps = interval * k + t_shift
            for t in range(from_dummy_steps, to_dummy_steps):
                if t < 0:
                    continue
                if t >= ic.depth - 1:
                    break
                for (i, j) in ic.ht.arcs:
                    for q in range(ic.num_lqubits):
                        if (problem.solution.get_values(ic.x_index(q, i, j, t)) > 0.5) and i < j:
                            mapped_dag.apply_operation_back(
                                op=SwapGate(),
                                qargs=[canonical_register[i], canonical_register[j]]
                            )
                            # update layout, swapping physical qubits (i, j)
                            # we cannot use Layout.swap() due to #virtuals < #physicals
                            v_org_i, v_org_j = None, None
                            if i in layout.get_physical_bits():
                                v_org_i = layout[i]
                            if j in layout.get_physical_bits():
                                v_org_j = layout[j]
                            if v_org_i is not None:
                                layout[v_org_i] = j
                            if v_org_j is not None:
                                layout[v_org_j] = i
            # map gates in k-th layer
            for node in layer['graph'].nodes():
                if node.type == 'op':
                    mapped_dag.apply_operation_back(
                        op=copy.deepcopy(node.op),
                        qargs=[canonical_register[layout[q]] for q in node.qargs],
                        cargs=node.cargs)
                # TODO: double check with y values?

        self.property_set['final_layout'] = copy.deepcopy(layout)

        return mapped_dag

    @staticmethod
    def _create_empty_dagcircuit(source_dag, canonical_qreg):
        target_dag = DAGCircuit()
        target_dag.name = source_dag.name
        target_dag._global_phase = source_dag._global_phase
        target_dag.metadata = source_dag.metadata

        target_dag.add_qubits(source_dag.qubits)
        target_dag.add_clbits(source_dag.clbits)

        target_dag.add_qreg(canonical_qreg)
        for creg in source_dag.cregs.values():
            target_dag.add_creg(creg)

        return target_dag
