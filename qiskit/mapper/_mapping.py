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

import networkx as nx
import numpy as np
import sympy
from sympy import Number as N

from qiskit.qasm import _node as node
from qiskit.mapper import MapperError
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit._dagcircuiterror import DAGCircuitError
from qiskit.unroll import DagUnroller, DAGBackend
from qiskit.mapper._quaternion import quaternion_from_euler

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

    # Find layout maximum index
    layout_max_index = max(map(lambda x: x[1]+1, layout.values()))

    # Can we already apply the gates?
    dist = sum([coupling.distance(layout[g[0]],
                                  layout[g[1]]) for g in gates])
    logger.debug("layer_permutation: dist = %s", dist)
    if dist == len(gates):
        logger.debug("layer_permutation: done already")
        logger.debug("layer_permutation: ----- exit -----")
        circ = DAGCircuit()
        circ.add_qreg('q', layout_max_index)
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
        trial_circ.add_qreg('q', layout_max_index)

        # Compute Sergey's randomized distance
        xi = {}
        for i in coupling.get_qubits():
            xi[i] = {}
        for i in coupling.get_qubits():
            for j in coupling.get_qubits():
                scale = 1 + np.random.normal(0, 1 / n)
                xi[i][j] = scale * coupling.distance(i, j) ** 2
                xi[j][i] = xi[i][j]

        # Loop over depths d up to a max depth of 2n+1
        d = 1
        # Circuit for this swap slice
        circ = DAGCircuit()
        circ.add_qreg('q', layout_max_index)
        circ.add_basis_element("CX", 2)
        circ.add_basis_element("cx", 2)
        circ.add_basis_element("swap", 2)
        circ.add_gate_data("cx", cx_data)
        circ.add_gate_data("swap", swap_data)

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
                for e in coupling.get_edges():
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
            dist = sum([coupling.distance(trial_layout[g[0]],
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
        dist = sum([coupling.distance(trial_layout[g[0]],
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


def direction_mapper(circuit_graph, coupling_graph):
    """Change the direction of CNOT gates to conform to CouplingGraph.

    circuit_graph = input DAGCircuit
    coupling_graph = corresponding CouplingGraph

    Adds "h" to the circuit basis.

    Returns a DAGCircuit object containing a circuit equivalent to
    circuit_graph but with CNOT gate directions matching the edges
    of coupling_graph. Raises an exception if the circuit_graph
    does not conform to the coupling_graph.
    """
    if "cx" not in circuit_graph.basis:
        return circuit_graph
    if circuit_graph.basis["cx"] != (2, 0, 0):
        raise MapperError("cx gate has unexpected signature %s" %
                          circuit_graph.basis["cx"])

    flipped_cx_circuit = DAGCircuit()
    flipped_cx_circuit.add_qreg('q', 2)
    flipped_cx_circuit.add_basis_element("CX", 2)
    flipped_cx_circuit.add_basis_element("U", 1, 0, 3)
    flipped_cx_circuit.add_basis_element("cx", 2)
    flipped_cx_circuit.add_basis_element("u2", 1, 0, 2)
    flipped_cx_circuit.add_basis_element("h", 1)
    flipped_cx_circuit.add_gate_data("cx", cx_data)
    flipped_cx_circuit.add_gate_data("u2", u2_data)
    flipped_cx_circuit.add_gate_data("h", h_data)
    flipped_cx_circuit.apply_operation_back("h", [("q", 0)])
    flipped_cx_circuit.apply_operation_back("h", [("q", 1)])
    flipped_cx_circuit.apply_operation_back("cx", [("q", 1), ("q", 0)])
    flipped_cx_circuit.apply_operation_back("h", [("q", 0)])
    flipped_cx_circuit.apply_operation_back("h", [("q", 1)])

    cg_edges = coupling_graph.get_edges()
    for cx_node in circuit_graph.get_named_nodes("cx"):
        nd = circuit_graph.multi_graph.node[cx_node]
        cxedge = tuple(nd["qargs"])
        if cxedge in cg_edges:
            logger.debug("cx %s[%d], %s[%d] -- OK",
                         cxedge[0][0], cxedge[0][1],
                         cxedge[1][0], cxedge[1][1])
            continue
        elif (cxedge[1], cxedge[0]) in cg_edges:
            circuit_graph.substitute_circuit_one(cx_node,
                                                 flipped_cx_circuit,
                                                 wires=[("q", 0), ("q", 1)])
            logger.debug("cx %s[%d], %s[%d] -FLIP",
                         cxedge[0][0], cxedge[0][1],
                         cxedge[1][0], cxedge[1][1])
        else:
            raise MapperError("circuit incompatible with CouplingGraph: "
                              "cx on %s" % pprint.pformat(cxedge))
    return circuit_graph


def swap_mapper_layer_update(i, first_layer, best_layout, best_d,
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
    layout_max_index = max(map(lambda x: x[1]+1, layout.values()))
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


def swap_mapper(circuit_graph, coupling_graph,
                initial_layout=None,
                basis="cx,u1,u2,u3,id", trials=20, seed=None):
    """Map a DAGCircuit onto a CouplingGraph using swap gates.

    Args:
        circuit_graph (DAGCircuit): input DAG circuit
        coupling_graph (CouplingGraph): coupling graph to map onto
        initial_layout (dict): dict from qubits of circuit_graph to qubits
            of coupling_graph (optional)
        basis (str): basis string specifying basis of output DAGCircuit
        trials (int): number of trials.
        seed (int): initial seed.

    Returns:
        DAGCircuit: object containing a circuit equivalent to
        circuit_graph that respects couplings in coupling_graph, and
        a layout dict mapping qubits of circuit_graph into qubits
        of coupling_graph. The layout may differ from the initial_layout
        if the first layer of gates cannot be executed on the
        initial_layout. Finally, returned is the final layer qubit
        permutation that is needed to add measurements back in.

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
        # Check the input layout
        circ_qubits = circuit_graph.get_qubits()
        coup_qubits = coupling_graph.get_qubits()
        qubit_subset = []
        for k, v in initial_layout.items():
            qubit_subset.append(v)
            if k not in circ_qubits:
                raise MapperError("initial_layout qubit %s[%d] not in input "
                                  "DAGCircuit" % (k[0], k[1]))
            if v not in coup_qubits:
                raise MapperError("initial_layout qubit %s[%d] not in input "
                                  "CouplingGraph" % (v[0], v[1]))
    else:
        # Supply a default layout
        qubit_subset = coupling_graph.get_qubits()
        qubit_subset = qubit_subset[0:circuit_graph.width()]
        initial_layout = {a: b for a, b in
                          zip(circuit_graph.get_qubits(), qubit_subset)}

    # Find swap circuit to preceed to each layer of input circuit
    layout = initial_layout.copy()
    layout_max_index = max(map(lambda x: x[1]+1, layout.values()))

    # Construct an empty DAGCircuit with one qreg "q"
    # and the same set of cregs as the input circuit
    dagcircuit_output = DAGCircuit()
    dagcircuit_output.name = circuit_graph.name
    dagcircuit_output.add_qreg("q", layout_max_index)
    for name, size in circuit_graph.cregs.items():
        dagcircuit_output.add_creg(name, size)

    # Make a trivial wire mapping between the subcircuits
    # returned by swap_mapper_layer_update and the circuit
    # we are building
    identity_wire_map = {}
    for j in range(layout_max_index):
        identity_wire_map[("q", j)] = ("q", j)
    for name, size in circuit_graph.cregs.items():
        for j in range(size):
            identity_wire_map[(name, j)] = (name, j)

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
                swap_mapper_layer_update(i,
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

    # This is the final layout that we need to correctly replace
    # any measurements that needed to be removed before the swap
    last_layout = layout

    # If first_layer is still set, the circuit only has single-qubit gates
    # so we can use the initial layout to output the entire circuit
    if first_layer:
        layout = initial_layout
        for i, layer in enumerate(layerlist):
            dagcircuit_output.compose_back(layer["graph"], layout)

    # Parse openqasm_output into DAGCircuit object
    dag_unrrolled = DagUnroller(dagcircuit_output,
                                DAGBackend(basis.split(",")))
    dagcircuit_output = dag_unrrolled.expand_gates()
    return dagcircuit_output, initial_layout, last_layout


def yzy_to_zyz(xi, theta1, theta2, eps=1e-9):
    """Express a Y.Z.Y single qubit gate as a Z.Y.Z gate.

    Solve the equation

    .. math::

    Ry(theta1).Rz(xi).Ry(theta2) = Rz(phi).Ry(theta).Rz(lambda)

    for theta, phi, and lambda.

    Return a solution theta, phi, and lambda.
    """
    Q = quaternion_from_euler([theta1, xi, theta2], 'yzy')
    euler = Q.to_zyz()
    P = quaternion_from_euler(euler, 'zyz')
    # output order different than rotation order
    out_angles = (euler[1], euler[0], euler[2])
    abs_inner = abs(P.data.dot(Q.data))
    if not np.allclose(abs_inner, 1, eps):
        logger.debug("xi=%s", xi)
        logger.debug("theta1=%s", theta1)
        logger.debug("theta2=%s", theta2)
        logger.debug("solutions=%s", out_angles)
        logger.debug("abs_inner=%s", abs_inner)
        raise MapperError('YZY and ZYZ angles do not give same rotation matrix.')
    return out_angles


def compose_u3(theta1, phi1, lambda1, theta2, phi2, lambda2):
    """Return a triple theta, phi, lambda for the product.

    u3(theta, phi, lambda)
       = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)
       = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)
       = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)
       = u3(theta', phi1 + phi', lambda2 + lambda')

    Return theta, phi, lambda.
    """
    # Careful with the factor of two in yzy_to_zyz
    thetap, phip, lambdap = yzy_to_zyz((lambda1 + phi2),
                                       theta1, theta2)
    (theta, phi, lamb) = (thetap, phi1 + phip, lambda2 + lambdap)
    return (theta, phi, lamb)


def cx_cancellation(circuit):
    """Cancel back-to-back "cx" gates in circuit."""
    runs = circuit.collect_runs(["cx"])
    for run in runs:
        # Partition the run into chunks with equal gate arguments
        partition = []
        chunk = []
        for i in range(len(run) - 1):
            chunk.append(run[i])
            qargs0 = circuit.multi_graph.node[run[i]]["qargs"]
            qargs1 = circuit.multi_graph.node[run[i + 1]]["qargs"]
            if qargs0 != qargs1:
                partition.append(chunk)
                chunk = []
        chunk.append(run[-1])
        partition.append(chunk)
        # Simplify each chunk in the partition
        for chunk in partition:
            if len(chunk) % 2 == 0:
                for n in chunk:
                    circuit._remove_op_node(n)
            else:
                for n in chunk[1:]:
                    circuit._remove_op_node(n)


def optimize_1q_gates(circuit):
    """Simplify runs of single qubit gates in the QX basis.

    Return a new circuit that has been optimized.
    """
    qx_basis = ["u1", "u2", "u3", "cx", "id"]
    dag_unroller = DagUnroller(circuit, DAGBackend(qx_basis))
    unrolled = dag_unroller.expand_gates()

    runs = unrolled.collect_runs(["u1", "u2", "u3", "id"])
    for run in runs:
        qname = unrolled.multi_graph.node[run[0]]["qargs"][0]
        right_name = "u1"
        right_parameters = (N(0), N(0), N(0))  # (theta, phi, lambda)
        for current_node in run:
            nd = unrolled.multi_graph.node[current_node]
            assert nd["condition"] is None, "internal error"
            assert len(nd["qargs"]) == 1, "internal error"
            assert nd["qargs"][0] == qname, "internal error"
            left_name = nd["name"]
            assert left_name in ["u1", "u2", "u3", "id"], "internal error"
            if left_name == "u1":
                left_parameters = (N(0), N(0), nd["params"][0])
            elif left_name == "u2":
                left_parameters = (sympy.pi / 2, nd["params"][0], nd["params"][1])
            elif left_name == "u3":
                left_parameters = tuple(nd["params"])
            else:
                left_name = "u1"  # replace id with u1
                left_parameters = (N(0), N(0), N(0))
            # Compose gates
            name_tuple = (left_name, right_name)
            if name_tuple == ("u1", "u1"):
                # u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
                right_parameters = (N(0), N(0), right_parameters[2] +
                                    left_parameters[2])
            elif name_tuple == ("u1", "u2"):
                # u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
                right_parameters = (sympy.pi / 2, right_parameters[1] +
                                    left_parameters[2], right_parameters[2])
            elif name_tuple == ("u2", "u1"):
                # u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
                right_name = "u2"
                right_parameters = (sympy.pi / 2, left_parameters[1],
                                    right_parameters[2] + left_parameters[2])
            elif name_tuple == ("u1", "u3"):
                # u1(lambda1) * u3(theta2, phi2, lambda2) =
                #     u3(theta2, phi2 + lambda1, lambda2)
                right_parameters = (right_parameters[0], right_parameters[1] +
                                    left_parameters[2], right_parameters[2])
            elif name_tuple == ("u3", "u1"):
                # u3(theta1, phi1, lambda1) * u1(lambda2) =
                #     u3(theta1, phi1, lambda1 + lambda2)
                right_name = "u3"
                right_parameters = (left_parameters[0], left_parameters[1],
                                    right_parameters[2] + left_parameters[2])
            elif name_tuple == ("u2", "u2"):
                # Using Ry(pi/2).Rz(2*lambda).Ry(pi/2) =
                #    Rz(pi/2).Ry(pi-2*lambda).Rz(pi/2),
                # u2(phi1, lambda1) * u2(phi2, lambda2) =
                #    u3(pi - lambda1 - phi2, phi1 + pi/2, lambda2 + pi/2)
                right_name = "u3"
                right_parameters = (sympy.pi - left_parameters[2] -
                                    right_parameters[1], left_parameters[1] +
                                    sympy.pi / 2, right_parameters[2] +
                                    sympy.pi / 2)
            elif name_tuple[1] == "nop":
                right_name = left_name
                right_parameters = left_parameters
            else:
                # For composing u3's or u2's with u3's, use
                # u2(phi, lambda) = u3(pi/2, phi, lambda)
                # together with the qiskit.mapper.compose_u3 method.
                right_name = "u3"
                # Evaluate the symbolic expressions for efficiency
                left_parameters = tuple(map(lambda x: x.evalf(), list(left_parameters)))
                right_parameters = tuple(map(lambda x: x.evalf(), list(right_parameters)))
                right_parameters = compose_u3(left_parameters[0],
                                              left_parameters[1],
                                              left_parameters[2],
                                              right_parameters[0],
                                              right_parameters[1],
                                              right_parameters[2])
                # Why evalf()? This program:
                #   OPENQASM 2.0;
                #   include "qelib1.inc";
                #   qreg q[2];
                #   creg c[2];
                #   u3(0.518016983430947*pi,1.37051598592907*pi,1.36816383603222*pi) q[0];
                #   u3(1.69867232277986*pi,0.371448347747471*pi,0.461117217930936*pi) q[0];
                #   u3(0.294319836336836*pi,0.450325871124225*pi,1.46804720442555*pi) q[0];
                #   measure q -> c;
                # took >630 seconds (did not complete) to optimize without
                # calling evalf() at all, 19 seconds to optimize calling
                # evalf() AFTER compose_u3, and 1 second to optimize
                # calling evalf() BEFORE compose_u3.
            # 1. Here down, when we simplify, we add f(theta) to lambda to
            # correct the global phase when f(theta) is 2*pi. This isn't
            # necessary but the other steps preserve the global phase, so
            # we continue in that manner.
            # 2. The final step will remove Z rotations by 2*pi.
            # 3. Note that is_zero is true only if the expression is exactly
            # zero. If the input expressions have already been evaluated
            # then these final simplifications will not occur.
            # TODO After we refactor, we should have separate passes for
            # exact and approximate rewriting.

            # Y rotation is 0 mod 2*pi, so the gate is a u1
            if (right_parameters[0] % (2 * sympy.pi)).is_zero \
                    and right_name != "u1":
                right_name = "u1"
                right_parameters = (0, 0, right_parameters[1] +
                                    right_parameters[2] +
                                    right_parameters[0])
            # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
            if right_name == "u3":
                # theta = pi/2 + 2*k*pi
                if ((right_parameters[0] - sympy.pi / 2) % (2 * sympy.pi)).is_zero:
                    right_name = "u2"
                    right_parameters = (sympy.pi / 2, right_parameters[1],
                                        right_parameters[2] +
                                        (right_parameters[0] - sympy.pi / 2))
                # theta = -pi/2 + 2*k*pi
                if ((right_parameters[0] + sympy.pi / 2) % (2 * sympy.pi)).is_zero:
                    right_name = "u2"
                    right_parameters = (sympy.pi / 2, right_parameters[1] +
                                        sympy.pi, right_parameters[2] -
                                        sympy.pi + (right_parameters[0] +
                                                    sympy.pi / 2))
            # u1 and lambda is 0 mod 2*pi so gate is nop (up to a global phase)
            if right_name == "u1" and (right_parameters[2] % (2 * sympy.pi)).is_zero:
                right_name = "nop"
            # Simplify the symbolic parameters
            right_parameters = tuple(map(sympy.simplify, list(right_parameters)))
        # Replace the data of the first node in the run
        new_params = []
        if right_name == "u1":
            new_params = [right_parameters[2]]
        if right_name == "u2":
            new_params = [right_parameters[1], right_parameters[2]]
        if right_name == "u3":
            new_params = list(right_parameters)

        nx.set_node_attributes(unrolled.multi_graph, name='name',
                               values={run[0]: right_name})
        # params is a list of sympy symbols
        nx.set_node_attributes(unrolled.multi_graph, name='params',
                               values={run[0]: new_params})
        # Delete the other nodes in the run
        for current_node in run[1:]:
            unrolled._remove_op_node(current_node)
        if right_name == "nop":
            unrolled._remove_op_node(run[0])
    return unrolled


def remove_last_measurements(dag_circuit, perform_remove=True):
    """Removes all measurements that occur as the last operation
    on a given qubit for a DAG circuit.  Measurements that are followed by
    additional gates are untouched.

    This operation is done in-place on the input DAG circuit if perform_pop=True.

    Parameters:
        dag_circuit (qiskit.dagcircuit._dagcircuit.DAGCircuit): DAG circuit.
        perform_remove (bool): Whether to perform removal, or just return node list.

    Returns:
        list: List of all measurements that were removed.
    """
    removed_meas = []
    try:
        meas_nodes = dag_circuit.get_named_nodes('measure')
    except DAGCircuitError:
        return removed_meas

    for idx in meas_nodes:
        _, succ_map = dag_circuit._make_pred_succ_maps(idx)
        if len(succ_map) == 2:
            # All succesors of the measurement are outputs, one for qubit and one for cbit
            # (As opposed to more gates being applied), and it is safe to remove the
            # measurement node and add it back after the swap mapper is done.
            removed_meas.append(dag_circuit.multi_graph.node[idx])
            if perform_remove:
                dag_circuit._remove_op_node(idx)
    return removed_meas


def return_last_measurements(dag_circuit, removed_meas, final_layout):
    """Returns the measurements to a quantum circuit, removed by
    `remove_last_measurements` after the swap mapper is finished.

    This operation is done in-place on the input DAG circuit.

    Parameters:
        dag_circuit (qiskit.dagcircuit._dagcircuit.DAGCircuit): DAG circuit.
        removed_meas (list): List of measurements previously removed.
        final_layout (dict): Qubit layout after swap mapping.
    """
    if any(removed_meas) and 'measure' not in dag_circuit.basis.keys():
        dag_circuit.add_basis_element("measure", 1, 1, 0)
    for meas in removed_meas:
        new_q_label = final_layout[meas['qargs'][0]]
        dag_circuit.apply_operation_back(name='measure', qargs=[new_q_label],
                                         cargs=meas['cargs'])
