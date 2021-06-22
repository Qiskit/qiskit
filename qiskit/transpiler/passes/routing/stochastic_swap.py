# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Map a DAGCircuit onto a `coupling_map` adding swap gates."""

import logging
from math import inf
from collections import OrderedDict
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout

# pylint: disable=no-name-in-module
from .cython.stochastic_swap.utils import nlayout_from_layout

# pylint: disable=no-name-in-module
from .cython.stochastic_swap.swap_trial import swap_trial


logger = logging.getLogger(__name__)


class StochasticSwap(TransformationPass):
    """Map a DAGCircuit onto a `coupling_map` adding swap gates.

    Uses a randomized algorithm.

    Notes:
        1. Measurements may occur and be followed by swaps that result in repeated
           measurement of the same qubit. Near-term experiments cannot implement
           these circuits, so some care is required when using this mapper
           with experimental backend targets.

        2. We do not use the fact that the input state is zero to simplify
           the circuit.
    """

    def __init__(self, coupling_map, trials=20, seed=None, fake_run=False):
        """StochasticSwap initializer.

        The coupling map is a connected graph

        If these are not satisfied, the behavior is undefined.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling
                map.
            trials (int): maximum number of iterations to attempt
            seed (int): seed for random number generator
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.trials = trials
        self.seed = seed
        self.fake_run = fake_run
        self.qregs = None
        self.rng = None
        self.trivial_layout = None
        self._qubit_indices = None

    def run(self, dag):
        """Run the StochasticSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("StochasticSwap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        canonical_register = dag.qregs["q"]
        self.trivial_layout = Layout.generate_trivial_layout(canonical_register)
        self._qubit_indices = {bit: idx for idx, bit in enumerate(dag.qubits)}

        self.qregs = dag.qregs
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.rng = np.random.default_rng(self.seed)
        logger.debug("StochasticSwap default_rng seeded with seed=%s", self.seed)

        new_dag = self._mapper(dag, self.coupling_map, trials=self.trials)
        return new_dag

    def _layer_permutation(self, layer_partition, layout, qubit_subset, coupling, trials):
        """Find a swap circuit that implements a permutation for this layer.

        The goal is to swap qubits such that qubits in the same two-qubit gates
        are adjacent.

        Based on S. Bravyi's algorithm.

        Args:
            layer_partition (list): The layer_partition is a list of (qu)bit
                lists and each qubit is a tuple (qreg, index).
            layout (Layout): The layout is a Layout object mapping virtual
                qubits in the input circuit to physical qubits in the coupling
                graph. It reflects the current positions of the data.
            qubit_subset (list): The qubit_subset is the set of qubits in
                the coupling graph that we have chosen to map into, as tuples
                (Register, index).
            coupling (CouplingMap): Directed graph representing a coupling map.
                This coupling map should be one that was provided to the
                stochastic mapper.
            trials (int): Number of attempts the randomized algorithm makes.

        Returns:
            Tuple: success_flag, best_circuit, best_depth, best_layout

        If success_flag is True, then best_circuit contains a DAGCircuit with
        the swap circuit, best_depth contains the depth of the swap circuit,
        and best_layout contains the new positions of the data qubits after the
        swap circuit has been applied.

        Raises:
            TranspilerError: if anything went wrong.
        """
        logger.debug("layer_permutation: layer_partition = %s", layer_partition)
        logger.debug("layer_permutation: layout = %s", layout.get_virtual_bits())
        logger.debug("layer_permutation: qubit_subset = %s", qubit_subset)
        logger.debug("layer_permutation: trials = %s", trials)

        # The input dag is on a flat canonical register
        # TODO: cleanup the code that is general for multiple qregs below
        canonical_register = QuantumRegister(len(layout), "q")
        qregs = OrderedDict({canonical_register.name: canonical_register})

        gates = []  # list of lists of tuples [[(register, index), ...], ...]
        for gate_args in layer_partition:
            if len(gate_args) > 2:
                raise TranspilerError("Layer contains > 2-qubit gates")
            if len(gate_args) == 2:
                gates.append(tuple(gate_args))
        logger.debug("layer_permutation: gates = %s", gates)

        # Can we already apply the gates? If so, there is no work to do.
        dist = sum(coupling.distance(layout[g[0]], layout[g[1]]) for g in gates)
        logger.debug("layer_permutation: distance = %s", dist)
        if dist == len(gates):
            logger.debug("layer_permutation: nothing to do")
            circ = DAGCircuit()
            circ.add_qreg(canonical_register)
            return True, circ, 0, layout

        # Begin loop over trials of randomized algorithm
        num_qubits = len(layout)
        best_depth = inf  # initialize best depth
        best_edges = None  # best edges found
        best_circuit = None  # initialize best swap circuit
        best_layout = None  # initialize best final layout

        cdist2 = coupling._dist_matrix ** 2
        # Scaling matrix
        scale = np.zeros((num_qubits, num_qubits))

        int_qubit_subset = np.fromiter(
            (self._qubit_indices[bit] for bit in qubit_subset),
            dtype=np.int32,
            count=len(qubit_subset),
        )

        int_gates = np.fromiter(
            (self._qubit_indices[bit] for gate in gates for bit in gate),
            dtype=np.int32,
            count=2 * len(gates),
        )

        int_layout = nlayout_from_layout(layout, self._qubit_indices, num_qubits, coupling.size())

        trial_circuit = DAGCircuit()  # SWAP circuit for slice of swaps in this trial
        trial_circuit.add_qubits(layout.get_virtual_bits())

        edges = np.asarray(coupling.get_edges(), dtype=np.int32).ravel()
        cdist = coupling._dist_matrix
        for trial in range(trials):
            logger.debug("layer_permutation: trial %s", trial)
            # This is one Trial --------------------------------------
            dist, optim_edges, trial_layout, depth_step = swap_trial(
                num_qubits,
                int_layout,
                int_qubit_subset,
                int_gates,
                cdist2,
                cdist,
                edges,
                scale,
                self.rng,
            )

            logger.debug("layer_permutation: final distance for this trial = %s", dist)
            if dist == len(gates) and depth_step < best_depth:
                logger.debug("layer_permutation: got circuit with improved depth %s", depth_step)
                best_edges = optim_edges
                best_layout = trial_layout
                best_depth = min(best_depth, depth_step)

            # Break out of trial loop if we found a depth 1 circuit
            # since we can't improve it further
            if best_depth == 1:
                break

        # If we have no best circuit for this layer, all of the
        # trials have failed
        if best_layout is None:
            logger.debug("layer_permutation: failed!")
            return False, None, None, None

        edges = best_edges.edges()
        for idx in range(best_edges.size // 2):
            swap_src = self.trivial_layout[edges[2 * idx]]
            swap_tgt = self.trivial_layout[edges[2 * idx + 1]]
            trial_circuit.apply_operation_back(SwapGate(), [swap_src, swap_tgt], [])
        best_circuit = trial_circuit

        # Otherwise, we return our result for this layer
        logger.debug("layer_permutation: success!")
        best_lay = best_layout.to_layout(qregs)
        return True, best_circuit, best_depth, best_lay

    def _layer_update(self, dag, layer, best_layout, best_depth, best_circuit):
        """Add swaps followed by the now mapped layer from the original circuit.

        Args:
            dag (DAGCircuit): The DAGCircuit object that the _mapper method is building
            layer (DAGCircuit): A DAGCircuit layer from the original circuit
            best_layout (Layout): layout returned from _layer_permutation
            best_depth (int): depth returned from _layer_permutation
            best_circuit (DAGCircuit): swap circuit returned from _layer_permutation
        """
        layout = best_layout
        logger.debug("layer_update: layout = %s", layout)
        logger.debug("layer_update: self.trivial_layout = %s", self.trivial_layout)

        # Output any swaps
        if best_depth > 0:
            logger.debug("layer_update: there are swaps in this layer, " "depth %d", best_depth)
            dag.compose(best_circuit)
        else:
            logger.debug("layer_update: there are no swaps in this layer")
        # Output this layer
        layer_circuit = layer["graph"]
        order = layout.reorder_bits(dag.qubits)
        dag.compose(layer_circuit, qubits=order)

    def _mapper(self, circuit_graph, coupling_graph, trials=20):
        """Map a DAGCircuit onto a CouplingMap using swap gates.

        Use self.trivial_layout for the initial layout.

        Args:
            circuit_graph (DAGCircuit): input DAG circuit
            coupling_graph (CouplingMap): coupling graph to map onto
            trials (int): number of trials.

        Returns:
            DAGCircuit: object containing a circuit equivalent to
                circuit_graph that respects couplings in coupling_graph

        Raises:
            TranspilerError: if there was any error during the mapping
                or with the parameters.
        """
        # Schedule the input circuit by calling layers()
        layerlist = list(circuit_graph.layers())
        logger.debug("schedule:")
        for i, v in enumerate(layerlist):
            logger.debug("    %d: %s", i, v["partition"])

        qubit_subset = self.trivial_layout.get_virtual_bits().keys()

        # Find swap circuit to precede each layer of input circuit
        layout = self.trivial_layout.copy()

        # Construct an empty DAGCircuit with the same set of
        # qregs and cregs as the input circuit
        dagcircuit_output = None
        if not self.fake_run:
            dagcircuit_output = circuit_graph._copy_circuit_metadata()

        logger.debug("trivial_layout = %s", layout)

        # Iterate over layers
        for i, layer in enumerate(layerlist):

            # Attempt to find a permutation for this layer
            success_flag, best_circuit, best_depth, best_layout = self._layer_permutation(
                layer["partition"], layout, qubit_subset, coupling_graph, trials
            )
            logger.debug("mapper: layer %d", i)
            logger.debug("mapper: success_flag=%s,best_depth=%s", success_flag, str(best_depth))

            # If this fails, try one gate at a time in this layer
            if not success_flag:
                logger.debug("mapper: failed, layer %d, " "retrying sequentially", i)
                serial_layerlist = list(layer["graph"].serial_layers())

                # Go through each gate in the layer
                for j, serial_layer in enumerate(serial_layerlist):

                    success_flag, best_circuit, best_depth, best_layout = self._layer_permutation(
                        serial_layer["partition"], layout, qubit_subset, coupling_graph, trials
                    )
                    logger.debug("mapper: layer %d, sublayer %d", i, j)
                    logger.debug(
                        "mapper: success_flag=%s,best_depth=%s,", success_flag, str(best_depth)
                    )

                    # Give up if we fail again
                    if not success_flag:
                        raise TranspilerError(
                            "swap mapper failed: " + "layer %d, sublayer %d" % (i, j)
                        )

                    # Update the record of qubit positions
                    # for each inner iteration
                    layout = best_layout
                    # Update the DAG
                    if not self.fake_run:
                        self._layer_update(
                            dagcircuit_output,
                            serial_layerlist[j],
                            best_layout,
                            best_depth,
                            best_circuit,
                        )

            else:
                # Update the record of qubit positions for each iteration
                layout = best_layout

                # Update the DAG
                if not self.fake_run:
                    self._layer_update(
                        dagcircuit_output, layerlist[i], best_layout, best_depth, best_circuit
                    )

        # This is the final edgemap. We might use it to correctly replace
        # any measurements that needed to be removed earlier.
        logger.debug("mapper: self.trivial_layout = %s", self.trivial_layout)
        logger.debug("mapper: layout = %s", layout)

        if self.fake_run:
            self.property_set["final_layout"] = layout
            return circuit_graph
        return dagcircuit_output
