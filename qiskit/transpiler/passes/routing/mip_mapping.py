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
import logging
import warnings

try:
    import cplex  # pylint: disable=unused-import
    _HAS_CPLEX = True
except ImportError:
    _HAS_CPLEX = False

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.transpiler import TransformationPass, CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.trivial_layout import TrivialLayout
from qiskit.transpiler.passes.routing.algorithms.mip_model import MIPMappingModel
from qiskit.transpiler.passmanager import PassManager

logger = logging.getLogger(__name__)


class MIPMapping(TransformationPass):
    """Map a DAGCircuit onto a `coupling_map` allocating qubits and adding swap gates.

    The MIP mapper try to find the best layout and routing at the same time
    by solving a mathematical optimization problem represented as a MIP (Mixed Integer Programming).
    """

    def __init__(self,
                 coupling_map,
                 objective="depth",
                 backend_prop=None,
                 time_limit=30):
        """MIPMapping initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            objective (str): Type of objective function; one of the following values.
                - error_rate: predicted error rate of the circuit
                - depth: [Default] depth (number of timesteps) of the circuit
                - balanced: weighted sum of error_rate and depth
            backend_prop (BackendProperties): Backend properties object
            time_limit (float): Time limit for solving MIP in seconds
        """
        if not _HAS_CPLEX:
            raise MissingOptionalLibraryError(
                libname="CPLEX",
                name="CplexOptimizer",
                pip_install="pip install 'qiskit[cplex]'",
            )
        super().__init__()
        if objective != "depth" and backend_prop is None:
            raise TranspilerError("'backend_prop' must be supplied when objective=='depth'")
        self.coupling_map = copy.deepcopy(coupling_map)  # save a copy since some methods modify it
        self.objective = objective
        self.backend_prop = backend_prop
        self.time_limit = time_limit

    def run(self, dag):
        """Run the MIPMapping pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if it fails to solve MIP problem within the time limit
        """
        if self.coupling_map is None:
            return dag

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError('More virtual qubits exist than physical qubits.')

        if self.property_set['layout']:
            logger.info("MIPMapping ignores given initial layout.")

        original_dag = dag
        # MIPMappingModel assumes num_virtual_qubits == num_physical_qubits
        # TODO: rewrite without dag<->circuit conversion (or remove the above assumption)
        pm = PassManager([
            TrivialLayout(self.coupling_map),
            FullAncillaAllocation(self.coupling_map),
            EnlargeWithAncilla()
        ])
        dag = circuit_to_dag(pm.run(dag_to_circuit(dag)))

        # TODO: set more safety dummy_steps
        dummy_steps = self.coupling_map.size() - 1
        # max_total_dummy = max_total_dummy or (self.dummy_steps * (self.dummy_steps - 1))

        model = MIPMappingModel(dag=dag,
                                coupling_map=self.coupling_map,
                                dummy_timesteps=dummy_steps)

        if len(model.su4layers) == 0:
            logger.info("MIPMapping is skipped due to no 2q-gates.")
            return original_dag

        model.create_cpx_problem(objective=self.objective)

        try:
            model.solve_cpx_problem(time_limit=self.time_limit)
        except TranspilerError as err:
            logger.warning("%s dag is not mapped in MIPMapping.", err.message)
            return original_dag

        # Get the optimized initial layout
        dic = {}
        for q in range(model.num_lqubits):
            for i in range(model.num_pqubits):
                if model.is_assigned(q, i, 0):
                    dic[model.index_to_virtual[q]] = i
        optimized_layout = Layout(dic)

        # Create a layout to track changes in layout for each layer
        layout = copy.deepcopy(optimized_layout)

        # Construct the mapped circuit
        canonical_register = QuantumRegister(self.coupling_map.size(), 'q')
        mapped_dag = self._create_empty_dagcircuit(dag, canonical_register)
        interval = dummy_steps + 1
        for k, layer in enumerate(dag.layers()):
            if model.is_su4layer(k):
                l = model.to_su4layer_depth(k)
                # add swaps between (l-1)-th and l-th su4layer
                from_steps = interval * (l-1)
                to_steps = interval * l
                for t in range(from_steps, to_steps):
                    if t < 0:
                        continue
                    if t >= model.depth - 1:
                        break
                    for (i, j) in model.edges:
                        for q in range(model.num_lqubits):
                            if model.is_swapped(q, i, j, t) and i < j:
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

        if self.property_set['layout'] and self.property_set['layout'] != optimized_layout:
            warnings.warn("MIPMapping changed the given initial layout", UserWarning)

        self.property_set['layout'] = optimized_layout
        self.property_set['final_layout'] = copy.deepcopy(layout)

        return mapped_dag

    @staticmethod
    def _create_empty_dagcircuit(source_dag, canonical_qreg):
        target_dag = DAGCircuit()
        target_dag.name = source_dag.name
        target_dag._global_phase = source_dag._global_phase
        target_dag.metadata = source_dag.metadata

        target_dag.add_qreg(canonical_qreg)
        for creg in source_dag.cregs.values():
            target_dag.add_creg(creg)

        return target_dag
