# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An analysis pass to find evolution gates in which the Paulis commute."""

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit import QuantumCircuit, QiskitError
from qiskit.opflow import PauliSumOp

from qiskit.quantum_info import Pauli

from qiskit.transpiler.passes.routing.swap_strategies.swap_strategy import SwapStrategy


class SwapStrategyRouter(TransformationPass):
    """An abstract base class to swap route one or more instructions to the coupling map.

    The mapping to the coupling map is done using swap strategies. Sub-classes must inherit
    from SwapStrategyRouter to define the type of instruction that they can handle and how to
    handle it.

    The swap strategy should suit the problem and the coupling map. This transpiler pass
    should ideally be executed before the quantum circuit is enlarged with any idle ancilla
    qubits. Otherwise we may swap qubits outside of the portion of the chip we want to use.
    Therefore, the swap strategy and its associated coupling map do not represent physical
    qubits. Instead, they represent an intermediate mapping that corresponds to the physical
    qubits once the initial layout is applied. The example below shows how to map a four
    qubit PauliEvolutionGate to qubits 0, 1, 3, and 4 of the five qubit device with the
    coupling map

    .. parsed-literal::

        0 -- 1 -- 2
             |
             3
             |
             4

    To do this we use a line swap strategy for qubits 0, 1, 3, and 4 defined it in terms
    of logical qubits 0, 1, 2, and 3.

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit.opflow import PauliSumOp
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.transpiler import Layout, CouplingMap, PassManager
        from qiskit.transpiler.passes import FullAncillaAllocation
        from qiskit.transpiler.passes import EnlargeWithAncilla
        from qiskit.transpiler.passes import ApplyLayout
        from qiskit.transpiler.passes import SetLayout

        from qiskit.transpiler.passes.routing.swap_strategies import (
            SwapStrategy,
            CheckCommutingEvolutions,
            PauliEvolutionGateRouter,
        )

        # Define the circuit on logical qubits
        op = PauliSumOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        # Define the swap strategy on qubits before the initial_layout is applied.
        swap_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(swap_cmap, swap_layers=[[(0, 1), (2, 3)], [(1, 2)]])

        # Chose qubits 0, 1, 3, and 4 from the backend coupling map shown above.
        backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4)])
        initial_layout = Layout.from_intlist([0, 1, 3, 4], *circ.qregs)

        pm_pre = PassManager(
            [
                CheckCommutingEvolutions(),
                PauliEvolutionGateRouter(swap_strat),
                SetLayout(initial_layout),
                FullAncillaAllocation(backend_cmap),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )

        # Insert swap gates, map to initial_layout and finally enlarge with ancilla.
        pm_pre.run(circ).draw("mpl")

    This pass manager relies on the ``current_layout`` which corresponds to the qubit layout as
    swap gates are applied. The pass will traverse all nodes in the dag. If a node should be
    routed using a swap strategy then it will be decomposed into sub-instructions with swap
    layers in between and the ``current_layout`` will be modified. Nodes that should not be
    routed using swap strategies will be added back to the dag taking the ``current_layout``
    into account.
    """

    # The node(s) that will be mapped must be of this type. Subclasses must specify this.
    __instruction_type__ = type(None)

    def __init__(self, swap_strategy: Optional[SwapStrategy] = None) -> None:
        """
        Args:
            swap_strategy: An instance of a SwapStrategy that holds the swap layers that
                are used, and the order in which to apply them, to map the instruction to
                the hardware. If this field is not given if should be contained in the
                property set of the pass. This allows other passes to determine the most
                appropriate swap strategy at run-time.
        """
        super().__init__()
        self._swap_strategy = swap_strategy

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass by decomposing the nodes it applies on.

        Args:
            dag: The dag to which we will add swaps.

        Returns:
            A dag where swaps have been added for the intended gate type.

        Raises:
            TranspilerError: If a node flagged for routing is not of the supported type.
            TranspilerError: If the swap strategy was not given at init time and there is
                no swap strategy in the property set.
        """
        if self._swap_strategy is None:
            swap_strategy = self.property_set["swap_strategy"]

            if swap_strategy is None:
                raise TranspilerError("No swap strategy given at init or in the property set.")
        else:
            swap_strategy = self._swap_strategy

        new_dag = self._empty_like(dag)

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        # Used to keep track of nodes that do not decompose using swap strategies.
        accumulator = self._empty_like(new_dag)

        for node in dag.topological_op_nodes():
            if node in self._get_nodes_to_decompose():

                if not isinstance(node.op, self.__instruction_type__):
                    raise TranspilerError(
                        f"{node.op} is not a {self.__instruction_type__.__name__}."
                    )

                # Check that the swap strategy creates enough connectivity for the node.
                self._check_edges(node, swap_strategy)

                # Compose any accumulated non-swap strategy gates to the dag
                accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)

        self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

        return new_dag

    def _compose_non_swap_nodes(
        self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit
    ) -> DAGCircuit:
        """Add all the non-swap strategy nodes that we have accumulated up to now.

        This method also resets the node accumulator to an empty dag.

        Args:
            layout: The current layout that keeps track of the swaps.
            new_dag: The new dag that we are building up.
            accumulator: A DAG to keep track of nodes that do not decompose
                using swap strategies.

        Returns:
            A new accumulator with the same registers as ``new_dag``.
        """
        # Add all the non-swap strategy nodes that we have accumulated up to now.
        order = layout.reorder_bits(new_dag.qubits)
        order_bis = [None] * len(layout)
        for idx, val in enumerate(order):
            order_bis[val] = idx

        new_dag.compose(accumulator, qubits=order_bis)

        # Re-initialize the node accumulator
        return self._empty_like(new_dag)

    @staticmethod
    def _empty_like(dag: DAGCircuit) -> DAGCircuit:
        """Create an empty dag with registers and metadata like the given dag."""
        new_dag = DAGCircuit()
        new_dag.name = dag.name
        new_dag.metadata = dag.metadata

        new_dag.add_qubits(dag.qubits)
        new_dag.add_clbits(dag.clbits)

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        return new_dag

    @staticmethod
    def _position_in_cmap(j: int, k: int, layout: Layout, dag: DAGCircuit) -> Tuple[int, ...]:
        """A helper function to track the movement of logical qubits through the swaps.

        Args:
            j: The index of decision variable j (i.e. logical qubit).
            k: The index of decision variable k (i.e. logical qubit).
            layout: The current layout that takes into account previous swap gates.
            dag: The dag.

        Returns:
            The position in the coupling map of the logical qubits j and k as a tuple.
        """
        pos_in_cmap = []
        for idx in range(len(layout)):
            if j == layout.get_virtual_bits()[dag.qubits[idx]]:
                pos_in_cmap.append(idx)
            if k == layout.get_virtual_bits()[dag.qubits[idx]]:
                pos_in_cmap.append(idx)

        return tuple(pos_in_cmap)

    @staticmethod
    def _build_sub_layers(current_layer: Dict[tuple, Gate]) -> List[Dict[tuple, Gate]]:
        """A helper method to build-up sets of gates that can simultaneously be applied.

        Note that this could also be done using an edge coloring of the coupling map.

        Args:
            current_layer: All gates in the current layer can be applied given the qubit ordering
            of the current layout. However, not all gates in the current layer can be applied
            simultaneously. This function creates sub-layers by greedily building up sub-layers
            of gates. All gates in a sub-layer can simultaneously be applied given the coupling
            map and current qubit configuration.

        Returns:
             A list of gate dicts that can be applied. The gates a position 0 are applied first.
             A gate dict has the qubit tuple as key and the gate to apply as value.
        """
        sub_layers = []
        while len(current_layer) > 0:
            current_sub_layer, remaining_gates, blocked_vertices = {}, {}, set()

            for edge, evo_gate in current_layer.items():
                if blocked_vertices.isdisjoint(edge):
                    current_sub_layer[edge] = evo_gate

                    # A vertex becomes blocked once a gate is applied to it.
                    blocked_vertices = blocked_vertices.union(edge)
                else:
                    remaining_gates[edge] = evo_gate

            current_layer = remaining_gates
            sub_layers.append(current_sub_layer)

        return sub_layers

    @abstractmethod
    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: SwapStrategy
    ) -> DAGCircuit:
        """Return a dag circuit of the decomposed node with swaps.

        This is the core method that sub-classes must implement. The method will decompose the
        node into a new dag that is compatible with the coupling map. Therefore, this is the
        part of the code that needs to insert the swap gates.

        Args:
            dag: The dag is needed to get access to qubit and other dag information.
            node: The node in the dag that we will decompose.
            current_layout: The current layout that is used to keep track of all the swaps.
                This layout should be modified in-place in this method.
            swap_strategy: The swap strategy used to decompose the node.

        Returns:
            A dag that is compatible with the coupling map where swap gates have been added
            to map the required gates to the hardware.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_nodes_to_decompose(self) -> List[DAGOpNode]:
        """Get the nodes that the pass will act on.

        Returns:
            A list of nodes in the dag to which this class will apply the :meth:`swap_decompose`
            method.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_edges(self, node: DAGOpNode, swap_strategy: SwapStrategy):
        """Check if the swap strategy can create the required connectivity.

        Args:
            node: The dag node for which to check if the swap strategy provides enough connectivity.
            swap_strategy: The swap strategy that is being used.

        Raises:
            TranspilerError: If there is an edge that the swap strategy cannot accommodate
                and if the pass has been configured to raise on such issues.
        """
        raise NotImplementedError


class PauliEvolutionGateRouter(SwapStrategyRouter):
    """A swap strategy pass that works for ``PauliEvolutionGate``s.

    Importantly, the Pauli gates in the evolution operator should commute for this
    pass to work properly.
    """

    __instruction_type__ = PauliEvolutionGate

    def __init__(self, swap_strategy: Optional[SwapStrategy] = None):
        """
        Args:
            swap_strategy: An instance of a SwapStrategy that holds the swap layers that
                are used, and the order in which to apply them, to map the instruction to
                the hardware. If this field is not given if should be contained in the
                property set of the pass. This allows other passes to determine the most
                appropriate swap strategy at run-time.
        """
        super().__init__(swap_strategy)

        # A dict set a run-time of required pauli terms. The key is a tuple of logical
        # qubit indices and the value is a tuple of a Pauli term and its coefficient.
        self._required_paulis = None

    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: SwapStrategy
    ) -> DAGCircuit:
        """Take an instance of ``PauliEvolutionGate`` and map it to the coupling map.

        The mapping is done with the swap strategy.

        Args:
            dag: The dag which contains the ``PauliEvolutionGate`` we route.
            node: A node whose operation is a ``PauliEvolutionGate``.
            current_layout: The layout before the swaps are applied. This function will
                modify the layout so that subsequent gates can be properly composed
                on the dag.
            swap_strategy: The swap strategy used to decompose the node.

        Returns:
            A dag that is compatible with the coupling map where swap gates have been added
            to map the ``PauliEvolutionGate`` to the hardware.
        """
        trivial_layout = Layout().generate_trivial_layout(*dag.qregs.values())
        gate_layers = self._layerize_op(dag, node.op, current_layout, swap_strategy)

        # Iterate over and apply gate layers
        max_distance = max(gate_layers.keys())

        circuit_with_swap = QuantumCircuit(len(*dag.qregs.values()))

        for i in range(max_distance + 1):
            # Get current layer and replace the problem indices j,k by the corresponding
            # positions in the coupling map. The current layer corresponds
            # to all the gates that can be applied at the ith swap layer.
            current_layer = {}
            for (j, k), evo_gate in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(j, k, current_layout, dag)] = evo_gate

            # Not all gates that are applied at the ith swap layer can be applied at the same
            # time. We therefore greedily build sub-layers.
            sub_layers = self._build_sub_layers(current_layer)

            # Apply sub-layers
            for sublayer in sub_layers:
                for edge, local_evo_gate in sublayer.items():
                    circuit_with_swap.append(local_evo_gate, edge)

            # Apply SWAP gates
            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]

                    circuit_with_swap.swap(j, k)
                    current_layout.swap(j, k)

        return circuit_to_dag(circuit_with_swap)

    @staticmethod
    def _pauli_to_edge(pauli: Pauli) -> Tuple[int, ...]:
        """Convert a pauli to an edge.

        Args:
            pauli: A pauli that is converted to a string to find out where non-identity
                Paulis are.

        Returns:
            A tuple representing where the Paulis are. For example, the Pauli "IZIZ" will
            return (0, 2) since logical qubits 0 and 2 interact.

        Raises:
            QiskitError: If the pauli does not exactly have two non-identity terms.
        """
        edge = tuple(i for i, p in enumerate(str(pauli)[::-1]) if p != "I")

        if len(edge) != 2:
            raise QiskitError(f"{pauli} does not have length two.")

        return edge

    def _check_edges(self, node: DAGOpNode, swap_strategy: SwapStrategy):
        """Check that the swap strategy can implement all edges needed by the node.

        Args:
            node: The node that contains the Pauli operator to check.
            swap_strategy: The swap strategy to specify the set of possible edges.

        Raises:
            TranspilerError: If there is an edge that the swap strategy cannot accommodate
                and if the pass has been configured to raise on such issues.
        """
        operator = node.op.operator

        self._required_paulis = {
            self._pauli_to_edge(pauli): (pauli, coeff)
            for pauli, coeff in zip(operator.paulis, operator.coeffs)
        }

        required_edges = set(self._required_paulis.keys())

        # Check that the swap strategy supports all required edges
        if not required_edges.issubset(swap_strategy.possible_edges):
            raise TranspilerError(
                f"{swap_strategy} cannot implement all edges in {required_edges}."
            )

    def _get_nodes_to_decompose(self) -> List[DAGOpNode]:
        """Get the nodes that the pass will act on.

        Returns:
            A list of nodes in the dag to which this class will apply the :meth:`swap_decompose`
            method.
        """
        return list(self.property_set.get("commuting_blocks", set()))

    def _layerize_op(
        self, dag: DAGCircuit, op: PauliSumOp, layout: Layout, swap_strategy: SwapStrategy
    ) -> Dict[int, Dict[tuple, PauliEvolutionGate]]:
        """Decompose the op into layers of terms depending on the swap strategy.

        This will classify the gates that we need to apply depending on when two logical
        qubits become adjacent in the swap strategy.

        Args:
            dag: The dag needed to get access to qubits.
            op: The operator with all the Pauli terms we need to apply.
            layout: The current layout before any swap gates have been applied.
            swap_strategy: The swap strategy from which to get the swap distance matrix.

        Returns:
            A dictionary where the key is an integer representing the swap layer at which
            the gate in the value can be applied on the coupling map because the qubits
            have become adjacent. The value is itself a dict where the key is a tuple
            representing the logical qubits on which the instruction in the value is applied.
            For example,

            .. parsed-literal::

                {
                    0: {
                        (2, 3): Instruction(name='exp(-i ZZ)', num_qubits=2, ...),
                        (0, 1): Instruction(name='exp(-i ZZ)', num_qubits=2, ...)
                    },
                    1: {
                        (0, 3): Instruction(name='exp(-i ZZ)', num_qubits=2, ...)
                    }
                }

            means that before any SWAP gates are applied we can apply ZZ gates on logical
            qubits (2, 3) and (0, 1). Then, after 1 layer of the swap strategy we can
            apply the gate between logical qubits 0 and 3 since they have become adjacent.

        Raises:
           QiskitError: If the Pauli coefficient has an iaginary part.
        """

        distance_matrix = swap_strategy.distance_matrix
        gate_layers = defaultdict(dict)

        for edge, (pauli, coeff) in self._required_paulis.items():

            bit0 = layout.get_virtual_bits()[dag.qubits[edge[0]]]
            bit1 = layout.get_virtual_bits()[dag.qubits[edge[1]]]

            distance = distance_matrix[bit0][bit1]

            simple_pauli = Pauli(str(pauli).replace("I", ""))

            if not np.isreal(coeff):
                raise QiskitError(f"Pauli {pauli} has a complex coefficient.")

            gate_layers[distance][edge] = PauliEvolutionGate(simple_pauli, op.time * np.real(coeff))

        return gate_layers
