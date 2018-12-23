# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, inconsistent-return-statements

"""
Layout module to assist with mapping circuit qubits onto physical qubits.
"""
import logging
import pprint
import sys

import numpy as np
import sympy

from qiskit.qasm import _node as node
from qiskit.mapper import MapperError
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister
from qiskit.extensions.standard.swap import SwapGate

logger = logging.getLogger(__name__)

# Notes:
# Measurements may occur and be followed by swaps that result in repeated
# measurement of the same qubit. Near-term experiments cannot implement
# these circuits, so we may need to modify the algorithm.
# It can happen that a swap in a deeper layer can be removed by permuting
# qubits in the layout. We don't do this.
# It can happen that initial swaps can be removed or partly simplified
# because the initial state is zero. We don't do this.

cx_data = {
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

swap_data = {
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

u2_data = {
    "opaque": False,
    "n_args": 2,
    "n_bits": 1,
    "args": ["phi", "lambda"],
    "bits": ["q"],
    # gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
    "body": node.GateBody([
        node.UniversalUnitary([
            node.ExpressionList([
                node.BinaryOp([
                    node.BinaryOperator('/'),
                    node.Real(sympy.pi),
                    node.Int(2)
                ]),
                node.Id("phi", 0, ""),
                node.Id("lambda", 0, "")
            ]),
            node.Id("q", 0, "")
        ])
    ])
}

h_data = {
    "opaque": False,
    "n_args": 0,
    "n_bits": 1,
    "args": [],
    "bits": ["a"],
    # gate h a { u2(0,pi) a; }
    "body": node.GateBody([
        node.CustomUnitary([
            node.Id("u2", 0, ""),
            node.ExpressionList([
                node.Int(0),
                node.Real(sympy.pi)
            ]),
            node.PrimaryList([
                node.Id("a", 0, "")
            ])
        ])
    ])
}


def layer_permutation(layer_partition, layout, qubit_subset, coupling, trials,
                      seed=None):
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
    if seed is not None:
        np.random.seed(seed)
    logger.debug("layer_permutation: ----- enter -----")
    logger.debug("layer_permutation: layer_partition = %s",
                 pprint.pformat(layer_partition))
    logger.debug("layer_permutation: layout = %s",
                 pprint.pformat(layout))
    logger.debug("layer_permutation: qubit_subset = %s",
                 pprint.pformat(qubit_subset))
    logger.debug("layer_permutation: trials = %s", trials)
    rev_layout = {b: a for a, b in layout.items()}
    gates = []
    for layer in layer_partition:
        if len(layer) > 2:
            raise MapperError("Layer contains >2 qubit gates")
        elif len(layer) == 2:
            gates.append(tuple(layer))

    logger.debug("layer_permutation: gates = %s", pprint.pformat(gates))

    # Can we already apply the gates?
    dist = sum([coupling.distance(layout[g[0]][1], layout[g[1]][1]) for g in gates])
    logger.debug("layer_permutation: dist = %s", dist)
    if dist == len(gates):
        logger.debug("layer_permutation: done already")
        logger.debug("layer_permutation: ----- exit -----")
        circ = DAGCircuit()
        circ.add_qreg(QuantumRegister(coupling.size(), "q"))
        circ.add_basis_element("CX", 2)
        circ.add_basis_element("cx", 2)
        circ.add_basis_element("swap", 2)
        circ.add_gate_data("cx", cx_data)
        circ.add_gate_data("swap", swap_data)
        return True, circ, 0, layout, bool(gates)

    # Begin loop over trials of randomized algorithm
    n = coupling.size()
    best_d = sys.maxsize  # initialize best depth
    best_circ = None  # initialize best swap circuit
    best_layout = None  # initialize best final layout
    for trial in range(trials):

        logger.debug("layer_permutation: trial %s", trial)
        trial_layout = layout.copy()
        rev_trial_layout = rev_layout.copy()
        # SWAP circuit constructed this trial
        trial_circ = DAGCircuit()
        trial_circ.add_qreg(QuantumRegister(coupling.size(), "q"))

        # Compute Sergey's randomized distance
        xi = {}
        for i in coupling.physical_qubits:
            xi[(QuantumRegister(coupling.size(), 'q'), i)] = {}
        for i in coupling.physical_qubits:
            i = (QuantumRegister(coupling.size(), 'q'), i)
            for j in coupling.physical_qubits:
                j = (QuantumRegister(coupling.size(), 'q'), j)
                scale = 1 + np.random.normal(0, 1 / n)
                xi[i][j] = scale * coupling.distance(i[1], j[1]) ** 2
                xi[j][i] = xi[i][j]

        # Loop over depths d up to a max depth of 2n+1
        d = 1
        # Circuit for this swap slice
        circ = DAGCircuit()
        circ.add_qreg(QuantumRegister(coupling.size(), "q"))
        circ.add_basis_element("CX", 2)
        circ.add_basis_element("cx", 2)
        circ.add_basis_element("swap", 2)
        circ.add_gate_data("cx", cx_data)
        circ.add_gate_data("swap", swap_data)

        # Identity wire-map for composing the circuits
        q = QuantumRegister(coupling.size(), 'q')
        identity_wire_map = {(q, j): (q, j) for j in range(coupling.size())}

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
                for e in coupling.get_edges():
                    e = [(QuantumRegister(coupling.size(), 'q'), edge) for edge in e]
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
                    circ.apply_operation_back(
                        SwapGate((opt_edge[0][0], opt_edge[0][1]),
                                 (opt_edge[1][0], opt_edge[1][1])))
                    logger.debug("layer_permutation: chose pair %s",
                                 pprint.pformat(opt_edge))
                else:
                    break

            # We have either run out of qubits or failed to improve
            # Compute the coupling graph distance_qubits
            dist = sum([coupling.distance(trial_layout[g[0]][1],
                                          trial_layout[g[1]][1]) for g in gates])
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
        dist = sum([coupling.distance(trial_layout[g[0]][1],
                                      trial_layout[g[1]][1]) for g in gates])
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


def swap_mapper_layer_update(i, first_layer, best_layout, best_d,
                             best_circ, layer_list, coupling_graph):
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
    dagcircuit_output.add_qreg(QuantumRegister(coupling_graph.size(), "q"))
    # Identity wire-map for composing the circuits
    q = QuantumRegister(coupling_graph.size(), 'q')
    identity_wire_map = {(q, j): (q, j) for j in range(coupling_graph.size())}

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


def swap_mapper(circuit_graph, coupling_graph,
                initial_layout=None, trials=20, seed=None):
    """Map a DAGCircuit onto a CouplingGraph using swap gates.

    Args:
        circuit_graph (DAGCircuit): input DAG circuit
        coupling_graph (CouplingGraph): coupling graph to map onto
        initial_layout (dict): dict {(str, int): (str, int)}
            from qubits of circuit_graph to qubits of coupling_graph (optional)
        trials (int): number of trials.
        seed (int): initial seed.

    Returns:
        DAGCircuit: object containing a circuit equivalent to
        circuit_graph that respects couplings in coupling_graph, and
        a layout dict mapping qubits of circuit_graph into qubits
        of coupling_graph. The layout may differ from the initial_layout
        if the first layer of gates cannot be executed on the
        initial_layout.

    Raises:
        MapperError: if there was any error during the mapping or with the
            parameters.
    """
    if circuit_graph.width() > coupling_graph.size():
        raise MapperError("Not enough qubits in CouplingGraph")

    # Schedule the input circuit
    layerlist = list(circuit_graph.layers())
    logger.debug("schedule:")
    for i, v in enumerate(layerlist):
        logger.debug("    %d: %s", i, v["partition"])

    if initial_layout is not None:
        # update initial_layout from a user given dict{(regname,idx): (regname,idx)}
        # to an expected dict{(reg,idx): (reg,idx)}
        device_register = QuantumRegister(coupling_graph.size(), 'q')
        initial_layout = {(circuit_graph.qregs[k[0]], k[1]): (device_register, v[1])
                          for k, v in initial_layout.items()}
        # Check the input layout
        circ_qubits = circuit_graph.get_qubits()
        coup_qubits = [(QuantumRegister(coupling_graph.size(), 'q'), wire) for wire in
                       coupling_graph.physical_qubits]
        qubit_subset = []
        for k, v in initial_layout.items():
            qubit_subset.append(v)
            if k not in circ_qubits:
                raise MapperError("initial_layout qubit %s[%d] not in input "
                                  "DAGCircuit" % (k[0].name, k[1]))
            if v not in coup_qubits:
                raise MapperError("initial_layout qubit %s[%d] not in input "
                                  "CouplingGraph" % (v[0].name, v[1]))
    else:
        # Supply a default layout
        qubit_subset = [(QuantumRegister(coupling_graph.size(), 'q'), wire) for wire in
                        coupling_graph.physical_qubits]
        qubit_subset = qubit_subset[0:circuit_graph.width()]
        initial_layout = {a: b for a, b in zip(circuit_graph.get_qubits(), qubit_subset)}

    # Find swap circuit to preceed to each layer of input circuit
    layout = initial_layout.copy()

    # Construct an empty DAGCircuit with one qreg "q"
    # and the same set of cregs as the input circuit
    dagcircuit_output = DAGCircuit()
    dagcircuit_output.name = circuit_graph.name
    dagcircuit_output.add_qreg(QuantumRegister(coupling_graph.size(), "q"))
    for creg in circuit_graph.cregs.values():
        dagcircuit_output.add_creg(creg)

    # Make a trivial wire mapping between the subcircuits
    # returned by swap_mapper_layer_update and the circuit
    # we are building
    identity_wire_map = {}
    q = QuantumRegister(coupling_graph.size(), 'q')
    for j in range(coupling_graph.size()):
        identity_wire_map[(q, j)] = (q, j)
    for creg in circuit_graph.cregs.values():
        for j in range(creg.size):
            identity_wire_map[(creg, j)] = (creg, j)

    first_layer = True  # True until first layer is output
    logger.debug("initial_layout = %s", layout)

    # Iterate over layers
    for i, layer in enumerate(layerlist):

        # Attempt to find a permutation for this layer
        success_flag, best_circ, best_d, best_layout, trivial_flag \
            = layer_permutation(layer["partition"], layout,
                                qubit_subset, coupling_graph, trials, seed)
        logger.debug("swap_mapper: layer %d", i)
        logger.debug("swap_mapper: success_flag=%s,best_d=%s,trivial_flag=%s",
                     success_flag, str(best_d), trivial_flag)

        # If this fails, try one gate at a time in this layer
        if not success_flag:
            logger.debug("swap_mapper: failed, layer %d, "
                         "retrying sequentially", i)
            serial_layerlist = list(layer["graph"].serial_layers())

            # Go through each gate in the layer
            for j, serial_layer in enumerate(serial_layerlist):

                success_flag, best_circ, best_d, best_layout, trivial_flag \
                    = layer_permutation(serial_layer["partition"],
                                        layout, qubit_subset, coupling_graph,
                                        trials, seed)
                logger.debug("swap_mapper: layer %d, sublayer %d", i, j)
                logger.debug("swap_mapper: success_flag=%s,best_d=%s,"
                             "trivial_flag=%s",
                             success_flag, str(best_d), trivial_flag)

                # Give up if we fail again
                if not success_flag:
                    raise MapperError("swap_mapper failed: " +
                                      "layer %d, sublayer %d" % (i, j) +
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
                    swap_mapper_layer_update(j,
                                             first_layer,
                                             best_layout,
                                             best_d,
                                             best_circ,
                                             serial_layerlist,
                                             coupling_graph),
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
                swap_mapper_layer_update(i,
                                         first_layer,
                                         best_layout,
                                         best_d,
                                         best_circ,
                                         layerlist,
                                         coupling_graph),
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

    return dagcircuit_output, initial_layout
