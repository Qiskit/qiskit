# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A pass implementing the default Qiskit stochastic mapper.
"""

from logging import getLogger
from pprint import pformat
from math import inf
import numpy as np

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.transpiler import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import SwapGate
from qiskit.mapper import Layout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements

logger = getLogger(__name__)


# Notes:
# 1. Measurements may occur and be followed by swaps that result in repeated
# measurement of the same qubit. Near-term experiments cannot implement
# these circuits, so some care is required when using this mapper
# with experimental backend targets.
# 2. We do not use the fact that the input state is zero to simplify
# the circuit.


class StochasticSwap(TransformationPass):
    """
    Maps a DAGCircuit onto a `coupling_map` adding swap gates.

    Uses a randomized algorithm.
    """

    def __init__(self, coupling_map, initial_layout=None,
                 trials=20, seed=None):
        """
        Map a DAGCircuit onto a `coupling_map` using swap gates.

        If initial_layout is not None, we assume the input circuit
        has been layed out before running this pass, and that
        the layout process yields a DAG, coupling map, and layout
        with the following properties:

        1. All three have the same number of qubits
        2. The layout a bijection from the DAG qubits to the coupling map

        For this mapping pass, it may also be necessary that

        3. The coupling map is a connected graph

        If these are not satisfied, the behavior is undefined.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling
                map.
            initial_layout (Layout): initial layout of qubits in mapping
            trials (int): maximum number of iterations to attempt
            seed (int): seed for random number generator
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.input_layout = initial_layout
        if initial_layout is not None:
            self.initial_layout = initial_layout.copy()
        else:
            self.initial_layout = None
        self.trials = trials
        self.seed = seed
        self.requires.append(BarrierBeforeFinalMeasurements())

    def run(self, dag):
        """
        Run the StochasticSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.
        """
        # If the property_set contains a layout, use it to
        # override any layout passed to __init__
        if self.property_set["layout"]:
            self.initial_layout = self.property_set["layout"]
            self.input_layout = self.property_set["layout"]
        new_dag = self._mapper(dag, self.coupling_map, trials=self.trials, seed=self.seed)
        # self.property_set["layout"] = self.initial_layout
        return new_dag

    def _layer_permutation(self, layer_partition, layout, qubit_subset,
                           coupling, trials, seed=None):
        """Find a swap circuit that implements a permutation for this layer.

        The goal is to swap qubits such that qubits in the same two-qubit gates
        are adjacent.

        Based on S. Bravyi's algorithm.

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
        seed (int): Optional seed for the random number generator. If it is
            None we do not reseed.

        Returns:
             Tuple: success_flag, best_circuit, best_depth, best_layout, trivial_flag

        If success_flag is True, then best_circuit contains a DAGCircuit with
        the swap circuit, best_depth contains the depth of the swap circuit,
        and best_layout contains the new positions of the data qubits after the
        swap circuit has been applied. The trivial_flag is set if the layer
        has no multi-qubit gates.

        Raises:
            TranspilerError: if anything went wrong.
        """
        if seed is not None:
            np.random.seed(seed)

        logger.debug("layer_permutation: layer_partition = %s",
                     pformat(layer_partition))
        logger.debug("layer_permutation: layout = %s",
                     pformat(layout.get_virtual_bits()))
        logger.debug("layer_permutation: qubit_subset = %s",
                     pformat(qubit_subset))
        logger.debug("layer_permutation: trials = %s", trials)

        gates = []  # list of lists of tuples [[(register, index), ...], ...]
        for gate_args in layer_partition:
            if len(gate_args) > 2:
                raise TranspilerError("Layer contains > 2-qubit gates")
            elif len(gate_args) == 2:
                gates.append(tuple(gate_args))
        logger.debug("layer_permutation: gates = %s", pformat(gates))

        # Can we already apply the gates? If so, there is no work to do.
        dist = sum([coupling.distance(layout[g[0]], layout[g[1]])
                    for g in gates])
        logger.debug("layer_permutation: distance = %s", dist)
        if dist == len(gates):
            logger.debug("layer_permutation: nothing to do")
            circ = DAGCircuit()
            for register in layout.get_virtual_bits().keys():
                if register[0] not in circ.qregs.values():
                    circ.add_qreg(register[0])
            circ.add_basis_element("swap", 2)
            return True, circ, 0, layout, (not bool(gates))

        # Begin loop over trials of randomized algorithm
        num_qubits = len(layout)
        best_depth = inf  # initialize best depth
        best_circuit = None  # initialize best swap circuit
        best_layout = None  # initialize best final layout
        for trial in range(trials):
            logger.debug("layer_permutation: trial %s", trial)
            trial_layout = layout.copy()
            trial_circuit = DAGCircuit()  # SWAP circuit for this trial
            for register in trial_layout.get_virtual_bits().keys():
                if register[0] not in trial_circuit.qregs.values():
                    trial_circuit.add_qreg(register[0])

            # Compute randomized distance
            xi = {}  # pylint: disable=invalid-name
            for i in range(num_qubits):
                xi[i] = {}
            for i in range(num_qubits):
                for j in range(i, num_qubits):
                    scale = 1 + np.random.normal(0, 1 / num_qubits)
                    xi[i][j] = scale * coupling.distance(i, j) ** 2
                    xi[j][i] = xi[i][j]

            slice_circuit = DAGCircuit()  # circuit for this swap slice
            for register in trial_layout.get_virtual_bits().keys():
                if register[0] not in slice_circuit.qregs.values():
                    slice_circuit.add_qreg(register[0])
            slice_circuit.add_basis_element("swap", 2)

            # Loop over depths from 1 up to a maximum depth
            depth_step = 1
            depth_max = 2 * num_qubits + 1
            while depth_step < depth_max:
                qubit_set = set(qubit_subset)
                # While there are still qubits available
                while qubit_set:
                    # Compute the objective function
                    min_cost = sum([xi[trial_layout[g[0]]][trial_layout[g[1]]] for g in gates])
                    # Try to decrease objective function
                    cost_reduced = False

                    # Loop over edges of coupling graph
                    for edge in coupling.get_edges():
                        qubits = [trial_layout[e] for e in edge]
                        # Are the qubits available?
                        if qubits[0] in qubit_set and qubits[1] in qubit_set:
                            # Try this edge to reduce the cost
                            new_layout = trial_layout.copy()
                            new_layout.swap(edge[0], edge[1])

                            # Compute the objective function
                            new_cost = sum([xi[new_layout[g[0]]][new_layout[g[1]]] for g in gates])
                            # Record progress if we succceed
                            if new_cost < min_cost:
                                logger.debug("layer_permutation: min_cost "
                                             "improved to %s", min_cost)
                                cost_reduced = True
                                min_cost = new_cost
                                optimal_layout = new_layout
                                optimal_edge = qubits

                    # Were there any good swap choices?
                    if cost_reduced:
                        qubit_set.remove(optimal_edge[0])
                        qubit_set.remove(optimal_edge[1])
                        trial_layout = optimal_layout
                        slice_circuit.apply_operation_back(
                            SwapGate(optimal_edge[0],
                                     optimal_edge[1]))
                        logger.debug("layer_permutation: swap the pair %s",
                                     pformat(optimal_edge))
                    else:
                        break

                # We have either run out of swap pairs to try or
                # failed to improve the cost.

                # Compute the coupling graph distance
                dist = sum([coupling.distance(trial_layout[g[0]],
                                              trial_layout[g[1]])
                            for g in gates])
                logger.debug("layer_permutation: new swap distance = %s", dist)
                # If all gates can be applied now, we are finished.
                # Otherwise we need to consider a deeper swap circuit
                if dist == len(gates):
                    logger.debug("layer_permutation: all gates can be "
                                 "applied now in this layer")
                    trial_circuit.extend_back(slice_circuit)
                    break

                # Increment the depth
                depth_step += 1
                logger.debug("layer_permutation: increment depth to %s", depth_step)

            # Either we have succeeded at some depth d < dmax or failed
            dist = sum([coupling.distance(trial_layout[g[0]],
                                          trial_layout[g[1]])
                        for g in gates])
            logger.debug("layer_permutation: final distance for this trial = %s", dist)
            if dist == len(gates):
                if depth_step < best_depth:
                    logger.debug("layer_permutation: got circuit with improved depth %s",
                                 depth_step)
                    best_circuit = trial_circuit
                    best_layout = trial_layout
                    best_depth = min(best_depth, depth_step)

            # Break out of trial loop if we found a depth 1 circuit
            # since we can't improve it further
            if best_depth == 1:
                break

        # If we have no best circuit for this layer, all of the
        # trials have failed
        if best_circuit is None:
            logger.debug("layer_permutation: failed!")
            return False, None, None, None, False

        # Otherwise, we return our result for this layer
        logger.debug("layer_permutation: success!")
        return True, best_circuit, best_depth, best_layout, False

    def _layer_update(self, i, first_layer, best_layout, best_depth,
                      best_circuit, layer_list):
        """Provide a DAGCircuit for a new mapped layer.

        i (int) = layer number
        first_layer (bool) = True if this is the first layer in the
            circuit with any multi-qubit gates
        best_layout (Layout) = layout returned from _layer_permutation
        best_depth (int) = depth returned from _layer_permutation
        best_circuit (DAGCircuit) = swap circuit returned
            from _layer_permutation
        layer_list (list) = list of DAGCircuit objects for each layer,
            output of DAGCircuit layers() method

        Return a DAGCircuit object to append to the output DAGCircuit
        that the _mapper method is building.
        """
        layout = best_layout
        logger.debug("layer_update: layout = %s", pformat(layout))
        logger.debug("layer_update: self.initial_layout = %s", pformat(self.initial_layout))
        dagcircuit_output = DAGCircuit()
        for register in layout.get_virtual_bits().keys():
            if register[0] not in dagcircuit_output.qregs.values():
                dagcircuit_output.add_qreg(register[0])

        # If this is the first layer with multi-qubit gates,
        # output all layers up to this point and ignore any
        # swap gates. Set the initial layout.
        if first_layer:
            logger.debug("layer_update: first multi-qubit gate layer")
            # Output all layers up to this point
            for j in range(i + 1):
                # Make qubit edge map and extend by classical bits
                edge_map = layout.combine_into_edge_map(self.initial_layout)
                for bit in dagcircuit_output.get_bits():
                    edge_map[bit] = bit
                dagcircuit_output.compose_back(layer_list[j]["graph"], edge_map)
        # Otherwise, we output the current layer and the associated swap gates.
        else:
            # Output any swaps
            if best_depth > 0:
                logger.debug("layer_update: there are swaps in this layer, "
                             "depth %d", best_depth)
                dagcircuit_output.extend_back(best_circuit)
            else:
                logger.debug("layer_update: there are no swaps in this layer")
            # Make qubit edge map and extend by classical bits
            edge_map = layout.combine_into_edge_map(self.initial_layout)
            for bit in dagcircuit_output.get_bits():
                edge_map[bit] = bit
            # Output this layer
            dagcircuit_output.compose_back(layer_list[i]["graph"], edge_map)

        return dagcircuit_output

    def _mapper(self, circuit_graph, coupling_graph,
                trials=20, seed=None):
        """Map a DAGCircuit onto a CouplingMap using swap gates.

        Use self.initial_layout for the initial layout.

        Args:
            circuit_graph (DAGCircuit): input DAG circuit
            coupling_graph (CouplingMap): coupling graph to map onto
            trials (int): number of trials.
            seed (int): initial seed.

        Returns:
            DAGCircuit: object containing a circuit equivalent to
                circuit_graph that respects couplings in coupling_graph
            Layout: a layout object mapping qubits of circuit_graph into
                qubits of coupling_graph. The layout may differ from the
                initial_layout if the first layer of gates cannot be
                executed on the initial_layout, since in this case
                it is more efficient to modify the layout instead of swapping
            Dict: a final-layer qubit permutation

        Raises:
            TranspilerError: if there was any error during the mapping
                or with the parameters.
        """
        # Schedule the input circuit by calling layers()
        layerlist = list(circuit_graph.layers())
        logger.debug("schedule:")
        for i, v in enumerate(layerlist):
            logger.debug("    %d: %s", i, v["partition"])

        if self.initial_layout is not None:
            qubit_subset = self.initial_layout.get_virtual_bits().keys()
        else:
            # Supply a default layout for this dag
            self.initial_layout = Layout()
            physical_qubit = 0
            for qreg in circuit_graph.qregs.values():
                for index in range(qreg.size):
                    self.initial_layout[(qreg, index)] = physical_qubit
                    physical_qubit += 1
            qubit_subset = self.initial_layout.get_virtual_bits().keys()
            # Restrict the coupling map to the image of the layout
            coupling_graph = coupling_graph.subgraph(
                self.initial_layout.get_physical_bits().keys())
            if coupling_graph.size() < len(self.initial_layout):
                raise TranspilerError("Coupling map too small for default layout")
            self.input_layout = self.initial_layout.copy()

        # Find swap circuit to preceed to each layer of input circuit
        layout = self.initial_layout.copy()

        # Construct an empty DAGCircuit with the same set of
        # qregs and cregs as the input circuit
        dagcircuit_output = DAGCircuit()
        dagcircuit_output.name = circuit_graph.name
        for qreg in circuit_graph.qregs.values():
            dagcircuit_output.add_qreg(qreg)
        for creg in circuit_graph.cregs.values():
            dagcircuit_output.add_creg(creg)

        # Make a trivial wire mapping between the subcircuits
        # returned by _layer_update and the circuit we build
        identity_wire_map = {}
        for qubit in circuit_graph.get_qubits():
            identity_wire_map[qubit] = qubit
        for bit in circuit_graph.get_bits():
            identity_wire_map[bit] = bit

        first_layer = True  # True until first layer is output
        logger.debug("initial_layout = %s", layout)

        # Iterate over layers
        for i, layer in enumerate(layerlist):

            # Attempt to find a permutation for this layer
            success_flag, best_circuit, best_depth, best_layout, trivial_flag \
                = self._layer_permutation(layer["partition"], layout,
                                          qubit_subset, coupling_graph,
                                          trials, seed)
            logger.debug("mapper: layer %d", i)
            logger.debug("mapper: success_flag=%s,best_depth=%s,trivial_flag=%s",
                         success_flag, str(best_depth), trivial_flag)

            # If this fails, try one gate at a time in this layer
            if not success_flag:
                logger.debug("mapper: failed, layer %d, "
                             "retrying sequentially", i)
                serial_layerlist = list(layer["graph"].serial_layers())

                # Go through each gate in the layer
                for j, serial_layer in enumerate(serial_layerlist):

                    success_flag, best_circuit, best_depth, best_layout, trivial_flag = \
                        self._layer_permutation(
                            serial_layer["partition"],
                            layout, qubit_subset,
                            coupling_graph,
                            trials, seed)
                    logger.debug("mapper: layer %d, sublayer %d", i, j)
                    logger.debug("mapper: success_flag=%s,best_depth=%s,"
                                 "trivial_flag=%s",
                                 success_flag, str(best_depth), trivial_flag)

                    # Give up if we fail again
                    if not success_flag:
                        raise TranspilerError("mapper failed: " +
                                              "layer %d, sublayer %d" %
                                              (i, j) + ", \"%s\"" %
                                              serial_layer["graph"].qasm(
                                                  no_decls=True,
                                                  aliases=layout))

                    # If this layer is only single-qubit gates,
                    # and we have yet to see multi-qubit gates,
                    # continue to the next inner iteration
                    if trivial_flag and first_layer:
                        logger.debug("mapper: skip to next sublayer")
                        continue

                    if first_layer:
                        self.initial_layout = layout

                    # Update the record of qubit positions
                    # for each inner iteration
                    layout = best_layout
                    # Update the DAG
                    dagcircuit_output.extend_back(
                        self._layer_update(j,
                                           first_layer,
                                           best_layout,
                                           best_depth,
                                           best_circuit,
                                           serial_layerlist),
                        identity_wire_map)
                    if first_layer:
                        first_layer = False

            else:
                # Update the record of qubit positions for each iteration
                layout = best_layout

                if first_layer:
                    self.initial_layout = layout

                # Update the DAG
                dagcircuit_output.extend_back(
                    self._layer_update(i,
                                       first_layer,
                                       best_layout,
                                       best_depth,
                                       best_circuit,
                                       layerlist),
                    identity_wire_map)

                if first_layer:
                    first_layer = False

        # This is the final edgemap. We might use it to correctly replace
        # any measurements that needed to be removed earlier.
        logger.debug("mapper: self.initial_layout = %s", pformat(self.initial_layout))
        logger.debug("mapper: layout = %s", pformat(layout))
        last_edgemap = layout.combine_into_edge_map(self.initial_layout)
        logger.debug("mapper: last_edgemap = %s", pformat(last_edgemap))

        # If first_layer is still set, the circuit only has single-qubit gates
        # so we can use the initial layout to output the entire circuit
        # This code is dead due to changes to first_layer above.
        if first_layer:
            logger.debug("mapper: first_layer flag still set")
            layout = self.initial_layout
            for i, layer in enumerate(layerlist):
                edge_map = layout.combine_into_edge_map(self.initial_layout)
                dagcircuit_output.compose_back(layer["graph"], edge_map)

        return dagcircuit_output
