# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Layout module to assist with mapping circuit qubits onto physical qubits.
"""
import sys
import copy
import math
import numpy as np
import networkx as nx
from ._mappererror import MapperError
from qiskit.qasm import Qasm
import qiskit.unroll as unroll

# Notes:
# Measurements may occur and be followed by swaps that result in repeated
# measurement of the same qubit. Near-term experiments cannot implement
# these circuits, so we may need to modify the algorithm.
# It can happen that a swap in a deeper layer can be removed by permuting
# qubits in the layout. We don't do this.
# It can happen that initial swaps can be removed or partly simplified
# because the initial state is zero. We don't do this.


def layer_permutation(layer_partition, layout, qubit_subset, coupling, trials, verbose=False):
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

    If success_flag is True, then best_circ contains an OPENQASM string with
    the swap circuit, best_d contains the depth of the swap circuit, and
    best_layout contains the new positions of the data qubits after the
    swap circuit has been applied. The trivial_flag is set if the layer
    has no multi-qubit gates.
    """
    if verbose:
        print("layer_permutation: ----- enter -----")
        print("layer_permutation: layer_partition = ", layer_partition)
        print("layer_permutation: layout = ", layout)
        print("layer_permutation: qubit_subset = ", qubit_subset)
        print("layer_permutation: trials = ", trials)
    rev_layout = {b: a for a, b in layout.items()}
    gates = []
    for layer in layer_partition:
        if len(layer) > 2:
            raise MapperError("Layer contains >2 qubit gates")
        elif len(layer) == 2:
            gates.append(tuple(layer))

    if verbose:
        print("layer_permutation: gates = ", gates)

    # Can we already apply the gates?
    dist = sum([coupling.distance(layout[g[0]],
                                  layout[g[1]]) for g in gates])
    if verbose:
        print("layer_permutation: dist = ", dist)
    if dist == len(gates):
        if verbose:
            print("layer_permutation: done already")
            print("layer_permutation: ----- exit -----")
        return True, "", 0, layout, len(gates) == 0

    # Begin loop over trials of randomized algorithm
    n = coupling.size()
    best_d = sys.maxsize  # initialize best depth
    best_circ = None  # initialize best swap circuit
    best_layout = None  # initialize best final layout
    for trial in range(trials):

        if verbose:
            print("layer_permutation: trial ", trial)
        trial_layout = copy.deepcopy(layout)
        rev_trial_layout = copy.deepcopy(rev_layout)
        trial_circ = ""  # circuit produced in this trial

        # Compute Sergey's randomized distance
        xi = {}
        for i in coupling.get_qubits():
            xi[i] = {}
        for i in coupling.get_qubits():
            for j in coupling.get_qubits():
                scale = 1.0 + np.random.normal(0.0, 1.0 / n)
                xi[i][j] = scale * coupling.distance(i, j)**2
                xi[j][i] = xi[i][j]

        # Loop over depths d up to a max depth of 2n+1
        d = 1
        circ = ""  # circuit for this swap slice
        while d < 2*n+1:
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
                        new_layout = copy.deepcopy(trial_layout)
                        new_layout[rev_trial_layout[e[0]]] = e[1]
                        new_layout[rev_trial_layout[e[1]]] = e[0]
                        rev_new_layout = copy.deepcopy(rev_trial_layout)
                        rev_new_layout[e[0]] = rev_trial_layout[e[1]]
                        rev_new_layout[e[1]] = rev_trial_layout[e[0]]
                        # Compute the objective function
                        new_cost = sum([xi[new_layout[g[0]]][new_layout[g[1]]]
                                        for g in gates])
                        # Record progress if we succceed
                        if new_cost < min_cost:
                            if verbose:
                                print("layer_permutation: progress! min_cost = ", min_cost)
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
                    circ += "swap %s[%d],%s[%d]; " % (opt_edge[0][0],
                                                      opt_edge[0][1],
                                                      opt_edge[1][0],
                                                      opt_edge[1][1])
                    if verbose:
                        print("layer_permutation: chose pair ", opt_edge)
                else:
                    break

            # We have either run out of qubits or failed to improve
            # Compute the coupling graph distance
            dist = sum([coupling.distance(trial_layout[g[0]],
                                          trial_layout[g[1]]) for g in gates])
            if verbose:
                print("layer_permutation: dist = ", dist)
            # If all gates can be applied now, we are finished
            # Otherwise we need to consider a deeper swap circuit
            if dist == len(gates):
                if verbose:
                    print("layer_permutation: all can be applied now")
                trial_circ += circ
                break

            # Increment the depth
            d += 1
            if verbose:
                print("layer_permutation: increment depth to ", d)

        # Either we have succeeded at some depth d < dmax or failed
        dist = sum([coupling.distance(trial_layout[g[0]],
                                      trial_layout[g[1]]) for g in gates])
        if verbose:
            print("layer_permutation: dist = ", dist)
        if dist == len(gates):
            if d < best_d:
                if verbose:
                    print("layer_permutation: got circuit with depth ", d)
                best_circ = trial_circ
                best_layout = trial_layout
            best_d = min(best_d, d)

    if best_circ is None:
        if verbose:
            print("layer_permutation: failed!")
            print("layer_permutation: ----- exit -----")
        return False, None, None, None, False
    else:
        if verbose:
            print("layer_permutation: done")
            print("layer_permutation: ----- exit -----")
        return True, best_circ, best_d, best_layout, False


def direction_mapper(circuit_graph, coupling_graph, verbose=False):
    """Change the direction of CNOT gates to conform to CouplingGraph.

    circuit_graph = input DAGCircuit
    coupling_graph = corresponding CouplingGraph
    verbose = optional flag to print more information

    Adds "h" to the circuit basis.

    Returns a DAGCircuit object containing a circuit equivalent to
    circuit_graph but with CNOT gate directions matching the edges
    of coupling_graph. Raises an exception if the circuit_graph
    does not conform to the coupling_graph.
    """
    if "cx" not in circuit_graph.basis:
        return circuit_graph
    if circuit_graph.basis["cx"] != (2, 0, 0):
        raise MapperError("cx gate has unexpected signature %s"
                              % circuit_graph.basis["cx"])
    flipped_qasm = "OPENQASM 2.0;\n" + \
                   "gate cx c,t { CX c,t; }\n" + \
                   "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }\n" + \
                   "gate h a { u2(0,pi) a; }\n" + \
                   "gate cx_flipped a,b { h a; h b; cx b, a; h a; h b; }\n" + \
                   "qreg q[2];\n" + \
                   "cx_flipped q[0],q[1];\n"
    u = unroll.Unroller(Qasm(data=flipped_qasm).parse(),
                        unroll.DAGBackend(["cx", "h"]))
    flipped_cx_circuit = u.execute()
    cx_node_list = circuit_graph.get_named_nodes("cx")
    cg_edges = coupling_graph.get_edges()
    for cx_node in cx_node_list:
        nd = circuit_graph.multi_graph.node[cx_node]
        cxedge = tuple(nd["qargs"])
        if cxedge in cg_edges:
            if verbose:
                print("cx %s[%d], %s[%d] -- OK" % (cxedge[0][0], cxedge[0][1],
                                                   cxedge[1][0], cxedge[1][1]))
            continue
        elif (cxedge[1], cxedge[0]) in cg_edges:
            circuit_graph.substitute_circuit_one(cx_node,
                                                 flipped_cx_circuit,
                                                 wires=[("q", 0), ("q", 1)])
            if verbose:
                print("cx %s[%d], %s[%d] -FLIP" % (cxedge[0][0], cxedge[0][1],
                                                   cxedge[1][0], cxedge[1][1]))
        else:
            raise MapperError("circuit incompatible with CouplingGraph: "
                                  + "cx on %s" % cxedge)
    return circuit_graph


def update_qasm(i, first_layer, best_layout, best_d,
                best_circ, circuit_graph, layer_list,
                verbose=False):
    """Update the QASM string for an iteration of swap_mapper.

    i = layer number
    first_layer = True if this is the first layer with multi-qubit gates
    best_layout = layout returned from swap algorithm
    best_d = depth returns from swap algorithm
    best_circ = swap circuit returned from swap algorithm
    circuit_graph = original input circuit
    layer_list = list of circuit objects for each layer
    verbose = set True for more print output

    Return openqasm_output, the QASM string to append.
    """
    openqasm_output = ""
    layout = best_layout

    # If this is the first layer with multi-qubit gates,
    # output all layers up to this point and ignore any
    # swap gates. Set the initial layout.
    if first_layer:
        if verbose:
            print("update_qasm_and_layout: first multi-qubit gate layer")
        # Output all layers up to this point
        openqasm_output += circuit_graph.qasm(
            add_swap=True,
            decls_only=True,
            aliases=layout)
        for j in range(i+1):
            openqasm_output += layer_list[j]["graph"].qasm(
                no_decls=True,
                aliases=layout)
    # Otherwise, we output the current layer and the associated swap gates.
    else:
        # Output any swaps
        if best_d > 0:
            if verbose:
                print("update_qasm_and_layout: swaps in this layer, depth %d" %
                      best_d)
            openqasm_output += best_circ
        else:
            if verbose:
                print("update_qasm_and_layout: no swaps in this layer")
        # Output this layer
        openqasm_output += layer_list[i]["graph"].qasm(
            no_decls=True,
            aliases=layout)
    return openqasm_output


def swap_mapper(circuit_graph, coupling_graph,
                initial_layout=None,
                basis="cx,u1,u2,u3,id", verbose=False, trials=20):
    """Map a DAGCircuit onto a CouplingGraph using swap gates.

    Args:
        circuit_graph (DAGCircuit): input DAG circuit
        coupling_graph (CouplingGraph): coupling graph to map onto
        initial_layout (dict): dict from qubits of circuit_graph to qubits
            of coupling_graph (optional)
        basis (str, optional): basis string specifying basis of output
            DAGCircuit
        verbose (bool, optional): print more information

    Returns:
        Returns a DAGCircuit object containing a circuit equivalent to
        circuit_graph that respects couplings in coupling_graph, and
        a layout dict mapping qubits of circuit_graph into qubits
        of coupling_graph. The layout may differ from the initial_layout
        if the first layer of gates cannot be executed on the
        initial_layout.
    """
    if circuit_graph.width() > coupling_graph.size():
        raise MapperError("Not enough qubits in CouplingGraph")

    # Schedule the input circuit
    layerlist = circuit_graph.layers()
    if verbose:
        print("schedule:")
        for i in range(len(layerlist)):
            print("    %d: %s" % (i, layerlist[i]["partition"]))

    if initial_layout is not None:
        # Check the input layout
        circ_qubits = circuit_graph.get_qubits()
        coup_qubits = coupling_graph.get_qubits()
        qubit_subset = []
        for k, v in initial_layout.items():
            qubit_subset.append(v)
            if k not in circ_qubits:
                raise MapperError("initial_layout qubit %s[%d] not " %
                                      (k[0], k[1]) +
                                      "in input DAGCircuit")
            if v not in coup_qubits:
                raise MapperError("initial_layout qubit %s[%d] not " %
                                      (v[0], v[1]) +
                                      " in input CouplingGraph")
    else:
        # Supply a default layout
        qubit_subset = coupling_graph.get_qubits()
        qubit_subset = qubit_subset[0:circuit_graph.width()]
        initial_layout = {a: b for a, b in
                          zip(circuit_graph.get_qubits(), qubit_subset)}

    # Find swap circuit to preceed to each layer of input circuit
    layout = copy.deepcopy(initial_layout)
    openqasm_output = ""
    first_layer = True  # True until first layer is output
    if verbose:
        print("initial_layout = ", layout)

    # Iterate over layers
    for i, layer in enumerate(layerlist):

        # Attempt to find a permutation for this layer
        success_flag, best_circ, best_d, best_layout, trivial_flag \
            = layer_permutation(layer["partition"], layout,
                                qubit_subset, coupling_graph, trials)
        if verbose:
            print("swap_mapper: layer %d" % i)
            print("swap_mapper: success_flag=%s," % success_flag +
                  "best_d=" + str(best_d) +
                  ",trivial_flag=%s" % trivial_flag)

        # If this layer is only single-qubit gates,
        # and we have yet to see multi-qubit gates,
        # continue to the next iteration
        if trivial_flag and first_layer:
            if verbose:
                print("swap_mapper: skip to next layer")
            continue

        # If this fails, try one gate at a time in this layer
        if not success_flag:
            if verbose:
                print("swap_mapper: failed, layer %d, " % i,
                      " retrying sequentially")
            serial_layerlist = layer["graph"].serial_layers()

            # Go through each gate in the layer
            for j, serial_layer in enumerate(serial_layerlist):

                success_flag, best_circ, best_d, best_layout, trivial_flag \
                    = layer_permutation(serial_layer["partition"],
                                        layout, qubit_subset, coupling_graph,
                                        trials)
                if verbose:
                    print("swap_mapper: layer %d, sublayer %d" % (i, j))
                    print("swap_mapper: success_flag=%s," % success_flag +
                          "best_d=" + str(best_d) +
                          ",trivial_flag=%s" % trivial_flag)

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
                    if verbose:
                        print("swap_mapper: skip to next sublayer")
                        continue

                # Update the record of qubit positions for each inner iteration
                layout = best_layout
                # Update the QASM
                openqasm_output += update_qasm(j, first_layer,
                                               best_layout, best_d,
                                               best_circ, circuit_graph,
                                               serial_layerlist, verbose)
                # Update initial layout
                if first_layer:
                    initial_layout = layout
                    first_layer = False

        else:
            # Update the record of qubit positions for each iteration
            layout = best_layout

            # Update the QASM
            openqasm_output += update_qasm(i, first_layer,
                                           best_layout, best_d,
                                           best_circ, circuit_graph,
                                           layerlist, verbose)
            # Update initial layout
            if first_layer:
                initial_layout = layout
                first_layer = False

    # If first_layer is still set, the circuit only has single-qubit gates
    # so we can use the initial layout to output the entire circuit
    if first_layer:
        layout = initial_layout
        openqasm_output += circuit_graph.qasm(
            add_swap=True,
            decls_only=True,
            aliases=layout)
        for i, layer in enumerate(layerlist):
            openqasm_output += layer["graph"].qasm(
                no_decls=True,
                aliases=layout)

    # Parse openqasm_output into DAGCircuit object
    basis += ",swap"
    ast = Qasm(data=openqasm_output).parse()
    u = unroll.Unroller(ast, unroll.DAGBackend(basis.split(",")))
    return u.execute(), initial_layout


def test_trig_solution(theta, phi, lamb, xi, theta1, theta2):
    """Test if arguments are a solution to a system of equations.

    .. math::
       \cos(\phi+\lambda) \cos(\\theta) = \cos(xi) * \cos(\\theta1+\\theta2)

       \sin(\phi+\lambda) \cos(\\theta) = \sin(xi) * \cos(\\theta1-\\theta2) 

       \cos(\phi-\lambda) \sin(\\theta) = \cos(xi) * \sin(\\theta1+\\theta2)

       \sin(\phi-\lambda) \sin(\\theta) = \sin(xi) * \sin(-\\theta1+\\theta2)

    Returns the maximum absolute difference between right and left hand sides.
    """
    delta1 = math.cos(phi + lamb) * math.cos(theta) - \
        math.cos(xi) * math.cos(theta1 + theta2)
    delta2 = math.sin(phi + lamb) * math.cos(theta) - \
        math.sin(xi) * math.cos(theta1 - theta2)
    delta3 = math.cos(phi - lamb) * math.sin(theta) - \
        math.cos(xi) * math.sin(theta1 + theta2)
    delta4 = math.sin(phi - lamb) * math.sin(theta) - \
        math.sin(xi) * math.sin(-theta1 + theta2)
    return max(map(abs, [delta1, delta2, delta3, delta4]))


def yzy_to_zyz(xi, theta1, theta2, eps=1e-9):
    """Express a Y.Z.Y single qubit gate as a Z.Y.Z gate.

    Solve the equation

    .. math::

    Ry(2*theta1).Rz(2*xi).Ry(2*theta2) = Rz(2*phi).Ry(2*theta).Rz(2*lambda)

    for theta, phi, and lambda. This is equivalent to solving the system
    given in the comment for test_solution. Use eps for comparisons with zero.

    Return a solution theta, phi, and lambda.
    """
    solutions = []  # list of potential solutions
    # Four cases to avoid singularities
    if abs(math.cos(xi)) < eps / 10:
        solutions.append((theta2 - theta1, xi, 0.0))
    elif abs(math.sin(theta1 + theta2)) < eps / 10:
        phi_minus_lambda = [
            math.pi / 2,
            3 * math.pi / 2,
            math.pi / 2,
            3 * math.pi / 2]
        stheta_1 = math.asin(math.sin(xi) * math.sin(-theta1 + theta2))
        stheta_2 = math.asin(-math.sin(xi) * math.sin(-theta1 + theta2))
        stheta_3 = math.pi - stheta_1
        stheta_4 = math.pi - stheta_2
        stheta = [stheta_1, stheta_2, stheta_3, stheta_4]
        phi_plus_lambda = list(map(lambda x:
                                   math.acos(math.cos(theta1 + theta2) *
                                             math.cos(xi) / math.cos(x)),
                                   stheta))
        sphi = [(term[0] + term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        slam = [(term[0] - term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        solutions = list(zip(stheta, sphi, slam))
    elif abs(math.cos(theta1 + theta2)) < eps / 10:
        phi_plus_lambda = [
            math.pi / 2,
            3 * math.pi / 2,
            math.pi / 2,
            3 * math.pi / 2]
        stheta_1 = math.acos(math.sin(xi) * math.cos(theta1 - theta2))
        stheta_2 = math.acos(-math.sin(xi) * math.cos(theta1 - theta2))
        stheta_3 = -stheta_1
        stheta_4 = -stheta_2
        stheta = [stheta_1, stheta_2, stheta_3, stheta_4]
        phi_minus_lambda = list(map(lambda x:
                                    math.acos(math.sin(theta1 + theta2) *
                                              math.cos(xi) / math.sin(x)),
                                    stheta))
        sphi = [(term[0] + term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        slam = [(term[0] - term[1]) / 2 for term in
                zip(phi_plus_lambda, phi_minus_lambda)]
        solutions = list(zip(stheta, sphi, slam))
    else:
        phi_plus_lambda = math.atan(math.sin(xi) * math.cos(theta1 - theta2) /
                                    (math.cos(xi) * math.cos(theta1 + theta2)))
        phi_minus_lambda = math.atan(math.sin(xi) * math.sin(-theta1 +
                                                             theta2) /
                                     (math.cos(xi) * math.sin(theta1 +
                                                              theta2)))
        sphi = (phi_plus_lambda + phi_minus_lambda) / 2
        slam = (phi_plus_lambda - phi_minus_lambda) / 2
        solutions.append((math.acos(math.cos(xi) * math.cos(theta1 + theta2) /
                                    math.cos(sphi + slam)), sphi, slam))
        solutions.append((math.acos(math.cos(xi) * math.cos(theta1 + theta2) /
                                    math.cos(sphi + slam + math.pi)),
                          sphi + math.pi / 2,
                          slam + math.pi / 2))
        solutions.append((math.acos(math.cos(xi) * math.cos(theta1 + theta2) /
                                    math.cos(sphi + slam)),
                          sphi + math.pi / 2, slam - math.pi / 2))
        solutions.append((math.acos(math.cos(xi) * math.cos(theta1 + theta2) /
                                    math.cos(sphi + slam + math.pi)),
                          sphi + math.pi, slam))
    # Select the first solution with the required accuracy
    deltas = list(map(lambda x: test_trig_solution(x[0], x[1], x[2],
                                                   xi, theta1, theta2),
                      solutions))
    for delta_sol in zip(deltas, solutions):
        if delta_sol[0] < eps:
            return delta_sol[1]
    print("xi=", xi)
    print("theta1=", theta1)
    print("theta2=", theta2)
    print("solutions=", solutions)
    print("deltas=", deltas)
    assert False, "Error! No solution found. This should not happen."


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
    thetap, phip, lambdap = yzy_to_zyz((lambda1 + phi2) / 2.0,
                                       theta1 / 2.0, theta2 / 2.0)
    return (2.0 * thetap, phi1 + 2.0 * phip, lambda2 + 2.0 * lambdap)


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
    urlr = unroll.Unroller(Qasm(data=circuit.qasm(qeflag=True)).parse(),
                           unroll.DAGBackend(qx_basis))
    unrolled = urlr.execute()

    runs = unrolled.collect_runs(["u1", "u2", "u3", "id"])
    for run in runs:
        qname = unrolled.multi_graph.node[run[0]]["qargs"][0]
        right_name = "u1"
        right_parameters = (0.0, 0.0, 0.0)  # (theta, phi, lambda)
        for node in run:
            nd = unrolled.multi_graph.node[node]
            assert nd["condition"] is None, "internal error"
            assert len(nd["qargs"]) == 1, "internal error"
            assert nd["qargs"][0] == qname, "internal error"
            left_name = nd["name"]
            assert left_name in ["u1", "u2", "u3", "id"], "internal error"
            if left_name == "u1":
                left_parameters = (0.0, 0.0, float(nd["params"][0]))
            elif left_name == "u2":
                left_parameters = (math.pi / 2, float(nd["params"][0]),
                                   float(nd["params"][1]))
            elif left_name == "u3":
                left_parameters = tuple(map(float, nd["params"]))
            else:
                left_name = "u1"  # replace id with u1
                left_parameters = (0.0, 0.0, 0.0)
            # Compose gates
            name_tuple = (left_name, right_name)
            if name_tuple == ("u1", "u1"):
                # u1(lambda1) * u1(lambda2) = u1(lambda1 + lambda2)
                right_parameters = (0.0, 0.0, right_parameters[2] +
                                    left_parameters[2])
            elif name_tuple == ("u1", "u2"):
                # u1(lambda1) * u2(phi2, lambda2) = u2(phi2 + lambda1, lambda2)
                right_parameters = (math.pi / 2, right_parameters[1] +
                                    left_parameters[2], right_parameters[2])
            elif name_tuple == ("u2", "u1"):
                # u2(phi1, lambda1) * u1(lambda2) = u2(phi1, lambda1 + lambda2)
                right_name = "u2"
                right_parameters = (math.pi / 2, left_parameters[1],
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
                right_parameters = (math.pi - left_parameters[2] -
                                    right_parameters[1], left_parameters[1] +
                                    math.pi / 2, right_parameters[2] +
                                    math.pi / 2)
            else:
                # For composing u3's or u2's with u3's, use
                # u2(phi, lambda) = u3(pi/2, phi, lambda)
                # together with the qiskit.mapper.compose_u3 method.
                right_name = "u3"
                right_parameters = compose_u3(left_parameters[0],
                                              left_parameters[1],
                                              left_parameters[2],
                                              right_parameters[0],
                                              right_parameters[1],
                                              right_parameters[2])
            # Here down, when we simplify, we add f(theta) to lambda to correct
            # the global phase when f(theta) is 2*pi. This isn't necessary but
            # the other steps preserve the global phase, so we continue.
            epsilon = 1e-9  # for comparison with zero
            # Y rotation is 0 mod 2*pi, so the gate is a u1
            if abs(right_parameters[0] % 2.0 * math.pi) < epsilon \
               and right_name != "u1":
                right_name = "u1"
                right_parameters = (0.0, 0.0, right_parameters[1] +
                                    right_parameters[2] +
                                    right_parameters[0])
            # Y rotation is pi/2 or -pi/2 mod 2*pi, so the gate is a u2
            if right_name == "u3":
                # theta = pi/2 + 2*k*pi
                if abs((right_parameters[0] - math.pi / 2) % 2.0 * math.pi) \
                   < epsilon:
                    right_name = "u2"
                    right_parameters = (math.pi / 2, right_parameters[1],
                                        right_parameters[2] +
                                        (right_parameters[0] - math.pi / 2))
                # theta = -pi/2 + 2*k*pi
                if abs((right_parameters[0] + math.pi / 2) % 2.0 * math.pi) \
                   < epsilon:
                    right_name = "u2"
                    right_parameters = (math.pi / 2, right_parameters[1] +
                                        math.pi, right_parameters[2] -
                                        math.pi + (right_parameters[0] +
                                                   math.pi / 2))
            # u1 and lambda is 0 mod 4*pi so gate is nop
            if right_name == "u1" and \
               abs(right_parameters[2] % 4.0 * math.pi) < epsilon:
                right_name = "nop"
        # Replace the data of the first node in the run
        new_params = []
        if right_name == "u1":
            new_params.append(right_parameters[2])
        if right_name == "u2":
            new_params = [right_parameters[1], right_parameters[2]]
        if right_name == "u3":
            new_params = list(right_parameters)
        nx.set_node_attributes(unrolled.multi_graph, 'name',
                               {run[0]: right_name})
        nx.set_node_attributes(unrolled.multi_graph, 'params',
                               {run[0]: tuple(map(str, new_params))})
        # Delete the other nodes in the run
        for node in run[1:]:
            unrolled._remove_op_node(node)
        if right_name == "nop":
            unrolled._remove_op_node(run[0])
    return unrolled
