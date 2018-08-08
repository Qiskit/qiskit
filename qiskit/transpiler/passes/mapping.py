# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name
"""
TODO: general description of what a mapper is
"""

import logging
import pprint
import sys

import numpy as np

from qiskit.transpiler._basepass import BasePass
from qiskit.mapper import (Coupling, MapperError, coupling_list2dict)
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm import _node as node
from qiskit.unroll import DagUnroller, DAGBackend

logger = logging.getLogger(__name__)


class SwapMapper(BasePass):
    """
    Map a DAGCircuit onto a CouplingGraph using swap gates.
    """

    def __init__(self, coupling_map=None, initial_layout=None, trials=20, seed=None):
        """
        Map a DAGCircuit onto a CouplingGraph using swap gates.
        Args:
            coupling_map (list): A graph of coupling::

            [
             [control0(int), target0(int)],
             [control1(int), target1(int)],
            ]

            eg. [[0, 2], [1, 2], [1, 3], [3, 4]}
            initial_layout (dict): A mapping of qubit to qubit::

                              {
                                ("q", start(int)): ("q", final(int)),
                                ...
                              }
                              eg.
                              {
                                ("q", 0): ("q", 0),
                                ("q", 1): ("q", 1),
                                ("q", 2): ("q", 2),
                                ("q", 3): ("q", 3)
                              }
            trials (int): number of trials.
            seed (int): initial seed.
        """
        super().__init__()
        self.coupling = Coupling(coupling_list2dict(coupling_map))
        self.last_layout = self.initial_layout = initial_layout
        self.trials = trials
        self.seed = seed
        self.cx_data = {
            "opaque": False,
            "n_args": 0,
            "n_bits": 2,
            "args": [],
            "bits": ["c", "t"],
            # gate cx c,t { CX c,t; }
            "body": node.GateBody([
                node.Cnot([
                    node.Id("c", 0, ""),
                    node.Id("t", 0, "")
                ])
            ])
        }

        self.swap_data = {
            "opaque": False,
            "n_args": 0,
            "n_bits": 2,
            "args": [],
            "bits": ["a", "b"],
            # gate swap a,b { cx a,b; cx b,a; cx a,b; }
            "body": node.GateBody([
                node.CustomUnitary([
                    node.Id("cx", 0, ""),
                    node.PrimaryList([
                        node.Id("a", 0, ""),
                        node.Id("b", 0, "")
                    ])
                ]),
                node.CustomUnitary([
                    node.Id("cx", 0, ""),
                    node.PrimaryList([
                        node.Id("b", 0, ""),
                        node.Id("a", 0, "")
                    ])
                ]),
                node.CustomUnitary([
                    node.Id("cx", 0, ""),
                    node.PrimaryList([
                        node.Id("a", 0, ""),
                        node.Id("b", 0, "")
                    ])
                ])
            ])
        }

    def run(self, dag):
        """
        Map a DAGCircuit onto a CouplingGraph using swap gates.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on

        Returns:
            DAGCircuit: the directed acyclic graph in which the pass was run.

        Raises:
            MapperError: if there was any error during the mapping or with the
                parameters.
        """
        if dag.width() > self.coupling.size():
            raise MapperError("Not enough qubits in CouplingGraph")

        # Schedule the input circuit
        layerlist = list(dag.layers())
        logger.debug("schedule:")
        for item, value in enumerate(layerlist):
            logger.debug("    %d: %s", item, value["partition"])

        if self.initial_layout is not None:
            # Check the input layout
            circ_qubits = dag.get_qubits()
            coup_qubits = self.coupling.get_qubits()
            qubit_subset = []
            for k, value in self.initial_layout.items():
                qubit_subset.append(value)
                if k not in circ_qubits:
                    raise MapperError("initial_layout qubit %s[%d] not in input "
                                      "DAGCircuit" % (k[0], k[1]))
                if value not in coup_qubits:
                    raise MapperError("initial_layout qubit %s[%d] not in input "
                                      "CouplingGraph" % (value[0], value[1]))
        else:
            # Supply a default layout
            qubit_subset = self.coupling.get_qubits()
            qubit_subset = qubit_subset[0:dag.width()]
            self.initial_layout = {a: b for a, b in
                                   zip(dag.get_qubits(), qubit_subset)}

        # Find swap circuit to preceed to each layer of input circuit
        layout = self.initial_layout.copy()
        layout_max_index = max(map(lambda x: x[1] + 1, layout.values()))

        # Construct an empty DAGCircuit with one qreg "q"
        # and the same set of cregs as the input circuit
        dagcircuit_output = DAGCircuit()
        dagcircuit_output.add_qreg("q", layout_max_index)
        for name, size in dag.cregs.items():
            dagcircuit_output.add_creg(name, size)

        # Make a trivial wire mapping between the subcircuits
        # returned by swap_mapper_layer_update and the circuit
        # we are building
        identity_wire_map = {}
        for j in range(layout_max_index):
            identity_wire_map[("q", j)] = ("q", j)
        for name, size in dag.cregs.items():
            for j in range(size):
                identity_wire_map[(name, j)] = (name, j)

        first_layer = True  # True until first layer is output
        logger.debug("initial_layout = %s", layout)

        # Iterate over layers
        for item, layer in enumerate(layerlist):

            # Attempt to find a permutation for this layer
            success_flag, best_circ, best_d, best_layout, trivial_flag \
                = self.layer_permutation(layer["partition"], layout, qubit_subset)
            logger.debug("swap_mapper: layer %d", item)
            logger.debug("swap_mapper: success_flag=%s,best_d=%s,trivial_flag=%s",
                         success_flag, str(best_d), trivial_flag)

            # If this fails, try one gate at a time in this layer
            if not success_flag:
                logger.debug("swap_mapper: failed, layer %d, "
                             "retrying sequentially", item)
                serial_layerlist = list(layer["graph"].serial_layers())

                # Go through each gate in the layer
                for j, serial_layer in enumerate(serial_layerlist):

                    success_flag, best_circ, best_d, best_layout, trivial_flag \
                        = self.layer_permutation(serial_layer["partition"], layout, qubit_subset)
                    logger.debug("swap_mapper: layer %d, sublayer %d", item, j)
                    logger.debug("swap_mapper: success_flag=%s,best_d=%s,"
                                 "trivial_flag=%s",
                                 success_flag, str(best_d), trivial_flag)

                    # Give up if we fail again
                    if not success_flag:
                        raise MapperError("swap_mapper failed: " +
                                          "layer %d, sublayer %d" % (item, j) +
                                          ", \"%s\"" %
                                          serial_layer["graph"].qasm(
                                              no_decls=True,
                                              aliases=layout))

                    # If this layer is only single-qubit gates,
                    # and we have yet to see multi-qubit gates,
                    # continue to the next inner iteration
                    if trivial_flag and first_layer:
                        logger.debug("swap_mapper: skip to next sublayer")
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
                        self.initial_layout = layout
                        first_layer = False

            else:
                # Update the record of qubit positions for each iteration
                layout = best_layout

                # Update the QASM
                dagcircuit_output.compose_back(
                    self.swap_mapper_layer_update(item,
                                                  first_layer,
                                                  best_layout,
                                                  best_d,
                                                  best_circ,
                                                  layerlist),
                    identity_wire_map)
                # Update initial layout
                if first_layer:
                    self.initial_layout = layout
                    first_layer = False

        # This is the final layout that we need to correctly replace
        # any measurements that needed to be removed before the swap
        self.last_layout = layout

        # If first_layer is still set, the circuit only has single-qubit gates
        # so we can use the initial layout to output the entire circuit
        if first_layer:
            layout = self.initial_layout
            for item, layer in enumerate(layerlist):
                dagcircuit_output.compose_back(layer["graph"], layout)

        dag_unrrolled = DagUnroller(dagcircuit_output,
                                    DAGBackend(self.shared_memory['basis']))
        dagcircuit_output = dag_unrrolled.expand_gates()

        self.shared_memory['final_layout'] = layout
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
        if self.seed is not None:
            np.random.seed(self.seed)
        logger.debug("layer_permutation: ----- enter -----")
        logger.debug("layer_permutation: layer_partition = %s",
                     pprint.pformat(layer_partition))
        logger.debug("layer_permutation: layout = %s",
                     pprint.pformat(layout))
        logger.debug("layer_permutation: qubit_subset = %s",
                     pprint.pformat(qubit_subset))
        logger.debug("layer_permutation: trials = %s", self.trials)
        rev_layout = {b: a for a, b in layout.items()}
        gates = []
        for layer in layer_partition:
            if len(layer) > 2:
                raise MapperError("Layer contains >2 qubit gates")
            elif len(layer) == 2:
                gates.append(tuple(layer))

        logger.debug("layer_permutation: gates = %s", pprint.pformat(gates))

        # Find layout maximum index
        layout_max_index = max(map(lambda x: x[1] + 1, layout.values()))

        # Can we already apply the gates?
        dist = sum([self.coupling.distance(layout[g[0]], layout[g[1]]) for g in gates])
        logger.debug("layer_permutation: dist = %s", dist)
        if dist == len(gates):
            logger.debug("layer_permutation: done already")
            logger.debug("layer_permutation: ----- exit -----")
            circ = DAGCircuit()
            circ.add_qreg('q', layout_max_index)
            circ.add_basis_element("CX", 2)
            circ.add_basis_element("cx", 2)
            circ.add_basis_element("swap", 2)
            circ.add_gate_data("cx", self.cx_data)
            circ.add_gate_data("swap", self.swap_data)
            return True, circ, 0, layout, bool(gates)

        # Begin loop over trials of randomized algorithm
        n = self.coupling.size()
        best_d = sys.maxsize  # initialize best depth
        best_circ = None  # initialize best swap circuit
        best_layout = None  # initialize best final layout
        for trial in range(self.trials):

            logger.debug("layer_permutation: trial %s", trial)
            trial_layout = layout.copy()
            rev_trial_layout = rev_layout.copy()
            # SWAP circuit constructed this trial
            trial_circ = DAGCircuit()
            trial_circ.add_qreg('q', layout_max_index)

            # Compute Sergey's randomized distance
            xi = {}
            for i in self.coupling.get_qubits():
                xi[i] = {}
            for i in self.coupling.get_qubits():
                for j in self.coupling.get_qubits():
                    scale = 1 + np.random.normal(0, 1 / n)
                    xi[i][j] = scale * self.coupling.distance(i, j) ** 2
                    xi[j][i] = xi[i][j]

            # Loop over depths d up to a max depth of 2n+1
            d = 1
            # Circuit for this swap slice
            circ = DAGCircuit()
            circ.add_qreg('q', layout_max_index)
            circ.add_basis_element("CX", 2)
            circ.add_basis_element("cx", 2)
            circ.add_basis_element("swap", 2)
            circ.add_gate_data("cx", self.cx_data)
            circ.add_gate_data("swap", self.swap_data)

            # Identity wire-map for composing the circuits
            identity_wire_map = {('q', j): ('q', j) for j in range(layout_max_index)}

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
                    for e in self.coupling.get_edges():
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
                                logger.debug("layer_permutation: progress! "
                                             "min_cost = %s", min_cost)
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
                        circ.apply_operation_back("swap", [(opt_edge[0][0],
                                                            opt_edge[0][1]),
                                                           (opt_edge[1][0],
                                                            opt_edge[1][1])])
                        logger.debug("layer_permutation: chose pair %s",
                                     pprint.pformat(opt_edge))
                    else:
                        break

                # We have either run out of qubits or failed to improve
                # Compute the coupling graph distance
                dist = sum([self.coupling.distance(trial_layout[g[0]],
                                                   trial_layout[g[1]]) for g in gates])
                logger.debug("layer_permutation: dist = %s", dist)
                # If all gates can be applied now, we are finished
                # Otherwise we need to consider a deeper swap circuit
                if dist == len(gates):
                    logger.debug("layer_permutation: all can be applied now")
                    trial_circ.compose_back(circ, identity_wire_map)
                    break

                # Increment the depth
                d += 1
                logger.debug("layer_permutation: increment depth to %s", d)

            # Either we have succeeded at some depth d < dmax or failed
            dist = sum([self.coupling.distance(trial_layout[g[0]],
                                               trial_layout[g[1]]) for g in gates])
            logger.debug("layer_permutation: dist = %s", dist)
            if dist == len(gates):
                if d < best_d:
                    logger.debug("layer_permutation: got circuit with depth %s", d)
                    best_circ = trial_circ
                    best_layout = trial_layout
                best_d = min(best_d, d)

        if best_circ is None:
            logger.debug("layer_permutation: failed!")
            logger.debug("layer_permutation: ----- exit -----")
            return False, None, None, None, False

        logger.debug("layer_permutation: done")
        logger.debug("layer_permutation: ----- exit -----")
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
        layout_max_index = max(map(lambda x: x[1] + 1, layout.values()))
        dagcircuit_output = DAGCircuit()
        dagcircuit_output.add_qreg("q", layout_max_index)
        # Identity wire-map for composing the circuits
        identity_wire_map = {('q', j): ('q', j) for j in range(layout_max_index)}

        # If this is the first layer with multi-qubit gates,
        # output all layers up to this point and ignore any
        # swap gates. Set the initial layout.
        if first_layer:
            logger.debug("update_qasm_and_layout: first multi-qubit gate layer")
            # Output all layers up to this point
            for j in range(i + 1):
                dagcircuit_output.compose_back(layer_list[j]["graph"], layout)
        # Otherwise, we output the current layer and the associated swap gates.
        else:
            # Output any swaps
            if best_d > 0:
                logger.debug("update_qasm_and_layout: swaps in this layer, "
                             "depth %d", best_d)
                dagcircuit_output.compose_back(best_circ, identity_wire_map)
            else:
                logger.debug("update_qasm_and_layout: no swaps in this layer")
            # Output this layer
            dagcircuit_output.compose_back(layer_list[i]["graph"], layout)
        return dagcircuit_output
