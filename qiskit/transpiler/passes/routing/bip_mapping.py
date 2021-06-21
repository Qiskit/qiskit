# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=import-error

"""Map a DAGCircuit onto a given ``coupling_map``, allocating qubits and adding swap gates."""
import copy
import logging
import math
import warnings

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.trivial_layout import TrivialLayout
from qiskit.transpiler.passes.routing.algorithms.bip_model import BIPMappingModel, HAS_CPLEX
from qiskit.transpiler.passmanager import PassManager

logger = logging.getLogger(__name__)


class BIPMapping(TransformationPass):
    r"""Map a DAGCircuit onto a given ``coupling_map``, allocating qubits and adding swap gates.

    The BIP mapper try to find the best layout and routing at once by
    solving a BIP (binary integer programming) problem as described in [1].

    The BIP problem represents the layer-by-layer mapping of 2-qubit gates, assuming all the gates
    in a layer can be run on the ``coupling_map``. In the problem, the variables :math:`w` represent
    the layout of qubits for each layer and the variables :math:`x` represent which pair fo qubits
    should be swapped inbetween layers. Based on the values in the solution of the BIP problem,
    the mapped circuit will be constructed.

    If you have circuits that need to be mapped and want to specify physical qubits (e.g. running
    Quantum Volume circuits), you have to specify ``coupling_map`` which contains only the qubits
    to be used. Do not use ``initial_layout`` for that purpose because the BIP mapper gracefully
    ignores ``initial_layout`` (and try to determines its best layout).

    **References:**

    [1] G. Nannicini et al. "Optimal qubit assignment and routing via integer programming."
    `arXiv:2106.06446 <https://arxiv.org/abs/2106.06446>`_
    """

    def __init__(
        self,
        coupling_map,
        objective="depth",
        backend_prop=None,
        time_limit=30,
        max_swaps_inbetween_layers=None,
    ):
        """BIPMapping initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            objective (str): Type of objective function:

                * ``'error_rate'``: [NotImplemented] Predicted error rate of the circuit
                * ``'depth'``: [Default] Depth (number of timesteps) of the circuit
                * ``'balanced'``: [NotImplemented] Weighted sum of ``'error_rate'`` and ``'depth'``

            backend_prop (BackendProperties): Backend properties object
            time_limit (float): Time limit for solving BIP in seconds
            max_swaps_inbetween_layers (int):
                Number of swaps allowed inbetween layers. If None, automatlically set.
                Large value could decrease the probability to build infeasible BIP problem but also
                could decrease the probability to find the optimal solution within the timelimit.

        Raises:
            MissingOptionalLibraryError: if cplex is not installed.
        """
        if not HAS_CPLEX:
            raise MissingOptionalLibraryError(
                libname="CPLEX",
                name="CplexOptimizer",
                pip_install="pip install 'qiskit[cplex]'",
            )
        super().__init__()
        self.coupling_map = copy.deepcopy(coupling_map)  # save a copy since some methods modify it
        self.objective = objective
        self.backend_prop = backend_prop
        self.time_limit = time_limit
        self.max_swaps_inbetween_layers = max_swaps_inbetween_layers

    def run(self, dag):
        """Run the BIPMapping pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if it fails to solve BIP problem within the time limit
        """
        if self.coupling_map is None:
            return dag

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical qubits.")

        if self.property_set["layout"]:
            logger.info("BIPMapping ignores given initial layout.")

        dummy_steps = math.ceil(math.sqrt(dag.num_qubits()))
        if self.max_swaps_inbetween_layers is not None:
            dummy_steps = max(0, self.max_swaps_inbetween_layers - 1)

        original_dag = dag
        # BIPMappingModel assumes num_virtual_qubits == num_physical_qubits
        # TODO: rewrite without dag<->circuit conversion (or remove the above assumption)
        pm = PassManager(
            [
                TrivialLayout(self.coupling_map),
                FullAncillaAllocation(self.coupling_map),
                EnlargeWithAncilla(),
            ]
        )
        dag = circuit_to_dag(pm.run(dag_to_circuit(dag)))

        model = BIPMappingModel(
            dag=dag, coupling_map=self.coupling_map, dummy_timesteps=dummy_steps
        )

        if len(model.su4layers) == 0:
            logger.info("BIPMapping is skipped due to no 2q-gates.")
            return original_dag

        model.create_cpx_problem(objective=self.objective)

        try:
            model.solve_cpx_problem(time_limit=self.time_limit)
        except TranspilerError as err:
            logger.warning("%s dag is not mapped in BIPMapping.", err.message)
            return original_dag

        # Get the optimized initial layout
        dic = {}
        for q in range(model.num_vqubits):
            for i in range(model.num_pqubits):
                if model.is_assigned(0, q, i):
                    dic[model.index_to_virtual[q]] = i
        optimized_layout = Layout(dic)

        # Create a layout to track changes in layout for each layer
        layout = copy.deepcopy(optimized_layout)

        # Construct the mapped circuit
        canonical_register = QuantumRegister(self.coupling_map.size(), "q")
        mapped_dag = self._create_empty_dagcircuit(dag, canonical_register)
        interval = dummy_steps + 1
        for k, layer in enumerate(dag.layers()):
            if model.is_su4layer(k):
                su4dep = model.to_su4layer_depth(k)
                # add swaps between (su4dep-1)-th and su4dep-th su4layer
                from_steps = interval * (su4dep - 1)
                to_steps = interval * su4dep
                for t in range(from_steps, to_steps):  # pylint: disable=invalid-name
                    if t < 0:
                        continue
                    if t >= model.depth - 1:
                        break
                    for (i, j) in model.arcs:
                        for q in range(model.num_vqubits):
                            if model.is_swapped(t, q, i, j) and i < j:
                                mapped_dag.apply_operation_back(
                                    op=SwapGate(),
                                    qargs=[canonical_register[i], canonical_register[j]],
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
            for node in layer["graph"].nodes():
                if node.type == "op":
                    mapped_dag.apply_operation_back(
                        op=copy.deepcopy(node.op),
                        qargs=[canonical_register[layout[q]] for q in node.qargs],
                        cargs=node.cargs,
                    )
                # TODO: double check with y values?

        if self.property_set["layout"] and self.property_set["layout"] != optimized_layout:
            warnings.warn("BIPMapping changed the given initial layout", UserWarning)

        self.property_set["layout"] = optimized_layout
        self.property_set["final_layout"] = copy.deepcopy(layout)

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
