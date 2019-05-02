# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
A pass implementing the legacy swapper.

Based on Sergey Bravyi's algorithm.
"""
import sys
import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit import QuantumRegister

from qiskit.extensions.standard import SwapGate
from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements


class LegacySwap(TransformationPass):
    """
    Maps a DAGCircuit onto a `coupling_map` adding swap gates.
    """

    def __init__(self,
                 coupling_map,
                 initial_layout=None,
                 trials=20,
                 seed=None):
        """
        Maps a DAGCircuit onto a `coupling_map` using swap gates.
        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            initial_layout (Layout): initial layout of qubits in mapping
            trials (int): the number of attempts the randomized algorithm makes.
            seed (int): initial seed.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout
        self.trials = trials
        self.seed = seed

    def run(self, dag):
        """Map a DAGCircuit onto a CouplingGraph using swap gates.

        Args:
            dag (DAGCircuit): input DAG circuit

        Returns:
            DAGCircuit: object containing a circuit equivalent to
            circuit_graph that respects couplings in coupling_map, and
            a layout dict mapping qubits of circuit_graph into qubits
            of coupling_map. The layout may differ from the initial_layout
            if the first layer of gates cannot be executed on the
            initial_layout.

        Raises:
            TranspilerError: if there was any error during the mapping or with the
                parameters.
        """
        if dag.width() > self.coupling_map.size():
            raise TranspilerError("Not enough qubits in CouplingGraph")

        # Schedule the input circuit
        layerlist = list(dag.layers())

        if self.initial_layout is None and self.property_set["layout"]:
            self.initial_layout = self.property_set["layout"]

        if self.initial_layout is not None:
            # update initial_layout from a user given dict{(regname,idx): (regname,idx)}
            # to an expected dict{(reg,idx): (reg,idx)}

            virtual_qubits = self.initial_layout.get_virtual_bits()
            self.initial_layout = {(v[0].name, v[1]): ('q', self.initial_layout[v]) for v in
                                   virtual_qubits}

            device_register = QuantumRegister(self.coupling_map.size(), 'q')
            initial_layout = {(dag.qregs[k[0]], k[1]): (device_register, v[1])
                              for k, v in self.initial_layout.items()}
            # Check the input layout
            circ_qubits = dag.qubits()
            coup_qubits = [(QuantumRegister(self.coupling_map.size(), 'q'), wire) for wire in
                           self.coupling_map.physical_qubits]
            qubit_subset = []
            for k, v in initial_layout.items():
                qubit_subset.append(v)
                if k not in circ_qubits:
                    raise TranspilerError("initial_layout qubit %s[%d] not in input "
                                          "DAGCircuit" % (k[0].name, k[1]))
                if v not in coup_qubits:
                    raise TranspilerError("initial_layout qubit %s[%d] not in input "
                                          "CouplingGraph" % (v[0].name, v[1]))
        else:
            # Supply a default layout
            qubit_subset = [(QuantumRegister(self.coupling_map.size(), 'q'), wire) for wire in
                            self.coupling_map.physical_qubits]
            qubit_subset = qubit_subset[0:dag.width()]
            initial_layout = {a: b for a, b in zip(dag.qubits(), qubit_subset)}

        # Find swap circuit to preceed to each layer of input circuit
        layout = initial_layout.copy()

        # Construct an empty DAGCircuit with one qreg "q"
        # and the same set of cregs as the input circuit
        dagcircuit_output = DAGCircuit()
        dagcircuit_output.name = dag.name
        dagcircuit_output.add_qreg(QuantumRegister(self.coupling_map.size(), "q"))
        for creg in dag.cregs.values():
            dagcircuit_output.add_creg(creg)

        # Make a trivial wire mapping between the subcircuits
        # returned by swap_mapper_layer_update and the circuit
        # we are building
        identity_wire_map = {}
        q = QuantumRegister(self.coupling_map.size(), 'q')
        for j in range(self.coupling_map.size()):
            identity_wire_map[(q, j)] = (q, j)
        for creg in dag.cregs.values():
            for j in range(creg.size):
                identity_wire_map[(creg, j)] = (creg, j)

        first_layer = True  # True until first layer is output

        # Iterate over layers
        for i, layer in enumerate(layerlist):

            # Attempt to find a permutation for this layer
            success_flag, best_circ, best_d, best_layout, trivial_flag \
                = self.layer_permutation(layer["partition"], layout, qubit_subset)

            # If this fails, try one gate at a time in this layer
            if not success_flag:
                serial_layerlist = list(layer["graph"].serial_layers())

                # Go through each gate in the layer
                for j, serial_layer in enumerate(serial_layerlist):

                    success_flag, best_circ, best_d, best_layout, trivial_flag \
                        = self.layer_permutation(serial_layer["partition"], layout, qubit_subset)

                    # Give up if we fail again
                    if not success_flag:
                        raise TranspilerError("swap_mapper failed: " +
                                              "layer %d, sublayer %d" % (i, j))

                    # If this layer is only single-qubit gates,
                    # and we have yet to see multi-qubit gates,
                    # continue to the next inner iteration
                    if trivial_flag and first_layer:
                        continue

                    # Update the record of qubit positions for each inner iteration
                    layout = best_layout
                    # Update the QASM
                    dagcircuit_output.compose_back(
                        self.swap_mapper_layer_update(j,
                                                      first_layer,
                                                      best_layout,
                                                      best_d,
                                                      best_circ,
                                                      serial_layerlist),
                        identity_wire_map)
                    # Update initial layout
                    if first_layer:
                        initial_layout = layout
                        first_layer = False

            else:
                # Update the record of qubit positions for each iteration
                layout = best_layout

                # Update the QASM
                dagcircuit_output.compose_back(
                    self.swap_mapper_layer_update(i,
                                                  first_layer,
                                                  best_layout,
                                                  best_d,
                                                  best_circ,
                                                  layerlist),
                    identity_wire_map)
                # Update initial layout
                if first_layer:
                    initial_layout = layout
                    first_layer = False

        # If first_layer is still set, the circuit only has single-qubit gates
        # so we can use the initial layout to output the entire circuit
        if first_layer:
            layout = initial_layout
            for i, layer in enumerate(layerlist):
                dagcircuit_output.compose_back(layer["graph"], layout)

        return dagcircuit_output

    def layer_permutation(self, layer_partition, layout, qubit_subset):
        """Find a swap circuit that implements a permutation for this layer.

        The goal is to swap qubits such that qubits in the same two-qubit gates
        are adjacent.

        Based on Sergey Bravyi's algorithm.

        The layer_partition is a list of (qu)bit lists and each qubit is a
        tuple (qreg, index).
        The layout is a dict mapping qubits in the circuit to qubits in the
        coupling graph and represents the current positions of the data.
        The qubit_subset is the subset of qubits in the coupling graph that
        we have chosen to map into.
        The coupling is a CouplingGraph.
        TRIALS is the number of attempts the randomized algorithm makes.

        Returns: success_flag, best_circ, best_d, best_layout, trivial_flag

        If success_flag is True, then best_circ contains a DAGCircuit with
        the swap circuit, best_d contains the depth of the swap circuit, and
        best_layout contains the new positions of the data qubits after the
        swap circuit has been applied. The trivial_flag is set if the layer
        has no multi-qubit gates.
        """
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.RandomState(self.seed)
        rev_layout = {b: a for a, b in layout.items()}
        gates = []
        for layer in layer_partition:
            if len(layer) > 2:
                raise TranspilerError("Layer contains >2 qubit gates")
            elif len(layer) == 2:
                gates.append(tuple(layer))

        # Can we already apply the gates?
        dist = sum([self.coupling_map.distance(layout[g[0]][1], layout[g[1]][1]) for g in gates])
        if dist == len(gates):
            circ = DAGCircuit()
            circ.add_qreg(QuantumRegister(self.coupling_map.size(), "q"))
            return True, circ, 0, layout, bool(gates)

        # Begin loop over trials of randomized algorithm
        n = self.coupling_map.size()
        best_d = sys.maxsize  # initialize best depth
        best_circ = None  # initialize best swap circuit
        best_layout = None  # initialize best final layout
        QR = QuantumRegister(self.coupling_map.size(), "q")
        for _ in range(self.trials):

            trial_layout = layout.copy()
            rev_trial_layout = rev_layout.copy()
            # SWAP circuit constructed this trial
            trial_circ = DAGCircuit()
            trial_circ.add_qreg(QR)

            # Compute Sergey's randomized distance
            xi = {}
            for i in self.coupling_map.physical_qubits:
                xi[(QR, i)] = {}
            for i in self.coupling_map.physical_qubits:
                i = (QR, i)
                for j in self.coupling_map.physical_qubits:
                    j = (QR, j)
                    scale = 1 + rng.normal(0, 1 / n)
                    xi[i][j] = scale * self.coupling_map.distance(i[1], j[1]) ** 2
                    xi[j][i] = xi[i][j]

            # Loop over depths d up to a max depth of 2n+1
            d = 1
            # Circuit for this swap slice
            circ = DAGCircuit()
            circ.add_qreg(QR)

            # Identity wire-map for composing the circuits
            identity_wire_map = {(QR, j): (QR, j) for j in range(n)}

            while d < 2 * n + 1:
                # Set of available qubits
                qubit_set = set(qubit_subset)
                # While there are still qubits available
                while qubit_set:
                    # Compute the objective function
                    min_cost = sum([xi[trial_layout[g[0]]][trial_layout[g[1]]]
                                    for g in gates])
                    # Try to decrease objective function
                    progress_made = False
                    # Loop over edges of coupling graph
                    for e in self.coupling_map.get_edges():
                        e = [(QR, edge) for edge in e]
                        # Are the qubits available?
                        if e[0] in qubit_set and e[1] in qubit_set:
                            # Try this edge to reduce the cost
                            new_layout = trial_layout.copy()
                            new_layout[rev_trial_layout[e[0]]] = e[1]
                            new_layout[rev_trial_layout[e[1]]] = e[0]
                            rev_new_layout = rev_trial_layout.copy()
                            rev_new_layout[e[0]] = rev_trial_layout[e[1]]
                            rev_new_layout[e[1]] = rev_trial_layout[e[0]]
                            # Compute the objective function
                            new_cost = sum([xi[new_layout[g[0]]][new_layout[g[1]]]
                                            for g in gates])
                            # Record progress if we succceed
                            if new_cost < min_cost:
                                progress_made = True
                                min_cost = new_cost
                                opt_layout = new_layout
                                rev_opt_layout = rev_new_layout
                                opt_edge = e

                    # Were there any good choices?
                    if progress_made:
                        qubit_set.remove(opt_edge[0])
                        qubit_set.remove(opt_edge[1])
                        trial_layout = opt_layout
                        rev_trial_layout = rev_opt_layout
                        circ.apply_operation_back(
                            SwapGate(),
                            [(opt_edge[0][0], opt_edge[0][1]), (opt_edge[1][0], opt_edge[1][1])],
                            [])
                    else:
                        break

                # We have either run out of qubits or failed to improve
                # Compute the coupling graph distance_qubits
                dist = sum([self.coupling_map.distance(trial_layout[g[0]][1],
                                                       trial_layout[g[1]][1]) for g in gates])
                # If all gates can be applied now, we are finished
                # Otherwise we need to consider a deeper swap circuit
                if dist == len(gates):
                    trial_circ.compose_back(circ, identity_wire_map)
                    break

                # Increment the depth
                d += 1

            # Either we have succeeded at some depth d < dmax or failed
            dist = sum([self.coupling_map.distance(trial_layout[g[0]][1],
                                                   trial_layout[g[1]][1]) for g in gates])
            if dist == len(gates):
                if d < best_d:
                    best_circ = trial_circ
                    best_layout = trial_layout
                best_d = min(best_d, d)

        if best_circ is None:
            return False, None, None, None, False

        return True, best_circ, best_d, best_layout, False

    def swap_mapper_layer_update(self, i, first_layer, best_layout, best_d,
                                 best_circ, layer_list):
        """Update the QASM string for an iteration of swap_mapper.

        i = layer number
        first_layer = True if this is the first layer with multi-qubit gates
        best_layout = layout returned from swap algorithm
        best_d = depth returned from swap algorithm
        best_circ = swap circuit returned from swap algorithm
        layer_list = list of circuit objects for each layer

        Return DAGCircuit object to append to the output DAGCircuit.
        """
        layout = best_layout
        dagcircuit_output = DAGCircuit()
        QR = QuantumRegister(self.coupling_map.size(), 'q')
        dagcircuit_output.add_qreg(QR)
        # Identity wire-map for composing the circuits
        identity_wire_map = {(QR, j): (QR, j) for j in range(self.coupling_map.size())}

        # If this is the first layer with multi-qubit gates,
        # output all layers up to this point and ignore any
        # swap gates. Set the initial layout.
        if first_layer:
            # Output all layers up to this point
            for j in range(i + 1):
                dagcircuit_output.compose_back(layer_list[j]["graph"], layout)
        # Otherwise, we output the current layer and the associated swap gates.
        else:
            # Output any swaps
            if best_d > 0:
                dagcircuit_output.compose_back(best_circ, identity_wire_map)

            # Output this layer
            dagcircuit_output.compose_back(layer_list[i]["graph"], layout)
        return dagcircuit_output
