"""
Layout module to assist with mapping circuit qubits onto physical qubits.

Author: Andrew Cross
"""
import sys
import copy
import numpy as np
from qiskit import QISKitException
from qiskit.qasm import Qasm
import qiskit.unroll as unroll


# TODO: might be simpler to go back to gate list form and do peephole stuff
# while we "play out" the gate sequence


def layer_permutation(layer_partition, layout, qubit_subset, coupling, trials):
    """Find a swap circuit that implements a permutation for this layer.

    The goal is to swap qubits such that qubits in the same two qubit gates
    are adjacent.

    Based on Sergey Bravyi's MATLAB code.

    The layer_partition is a list of (qu)bit lists and each qubit is a
    tuple (qreg, index).
    The layout is a dict mapping qubits in the circuit to qubits in the
    coupling graph and represents the current positions of the data.
    The qubit_subset is the subset of qubits in the coupling graph that
    we have chosen to map into.
    The coupling is a CouplingGraph.
    TRIALS is the number of attempts the randomized algorithm makes.

    Returns: success_flag, best_circ, best_d, best_layout

    If success_flag is True, then best_circ contains an OPENQASM string with
    the swap circuit, best_d contains the depth of the swap circuit, and
    best_layout contains the new positions of the data qubits after the
    swap circuit has been applied.
    """
    rev_layout = {b: a for a, b in layout.items()}
    gates = []
    for layer in layer_partition:
        if len(layer) > 2:
            raise QISKitException("Layer contains >2 qubit gates")
        elif len(layer) == 2:
            gates.append(tuple(layer))

    # Begin loop over trials of randomized algorithm
    n = coupling.size()
    best_d = sys.maxsize  # initialize best depth
    best_circ = None  # initialize best swap circuit
    best_layout = None  # initialize best final layout
    for trial in range(trials):

        trial_layout = copy.deepcopy(layout)
        rev_trial_layout = copy.deepcopy(rev_layout)
        trial_circ = ""

        # Compute Sergey's randomized distance
        xi = {}
        for i in coupling.get_qubits():
            xi[i] = {}
        for i in coupling.get_qubits():
            for j in coupling.get_qubits():
                scale = 1.0 + np.random.normal(0.0, 1.0/n)
                xi[i][j] = scale * coupling.distance(i, j)**2
                xi[j][i] = xi[i][j]

        # Loop over depths d up to a max depth of 2n+1
        for d in range(1, 2*n+1):
            circ = ""
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
                else:
                    break

            # We have either run out of qubits or failed to improve
            # Compute the coupling graph distance
            dist = sum([coupling.distance(trial_layout[g[0]],
                                          trial_layout[g[1]]) for g in gates])
            # If all gates can be applied now, we are finished
            # Otherwise we need to consider a deeper swap circuit
            if dist == len(gates):
                trial_circ = circ
                break

        # Either we have succeeded at some depth d < dmax or failed
        dist = sum([coupling.distance(trial_layout[g[0]],
                                      trial_layout[g[1]]) for g in gates])
        if dist == len(gates):
            if d < best_d:
                best_circ = trial_circ
                best_layout = trial_layout
            best_d = min(best_d, d)

    if best_circ is None:
        return False, None, None, None
    else:
        return True, best_circ, best_d, best_layout


def direction_mapper(circuit_graph, coupling_graph, verbose=False):
    """Change the direction of CNOT gates to conform to CouplingGraph.

    circuit_graph = input Circuit
    coupling_graph = corresponding CouplingGraph
    verbose = optional flag to print more information

    Adds "h" to the circuit basis.

    Returns a Circuit object containing a circuit equivalent to
    circuit_graph but with CNOT gate directions matching the edges
    of coupling_graph. Raises an exception if the circuit_graph
    does not conform to the coupling_graph.
    """
    if "cx" not in circuit_graph.basis:
        return circuit_graph
    if circuit_graph.basis["cx"] != (2, 0, 0):
        raise QISKitException("cx gate has unexpected signature %s"
                              % circuit_graph.basis["cx"])
    flipped_qasm = "OPENQASM 2.0;\n" + \
                   "gate cx c,t { CX c,t; }\n" + \
                   "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }\n" + \
                   "gate h a { u2(0,pi) a; }\n" + \
                   "gate cx_flipped a,b { h a; h b; cx b, a; h a; h b; }\n" + \
                   "qreg q[2];\n" + \
                   "cx_flipped q[0],q[1];\n"
    u = unroll.Unroller(Qasm(data=flipped_qasm).parse(),
                        unroll.CircuitBackend(["cx", "h"]))
    u.execute()
    flipped_cx_circuit = u.backend.circuit
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
            raise QISKitException("circuit incompatible with CouplingGraph: "
                                  + "cx on %s" % cxedge)
    return circuit_graph


def swap_mapper(circuit_graph, coupling_graph,
                initial_layout=None,
                basis="cx,u1,u2,u3", verbose=False):
    """Map a Circuit onto a CouplingGraph using swap gates.

    circuit_graph = input Circuit
    coupling_graph = CouplingGraph to map onto
    initial_layout = dict from qubits of circuit_graph to qubits
      of coupling_graph (optional)
    basis = basis string specifying basis of output Circuit
    verbose = optional flag to print more information

    Returns a Circuit object containing a circuit equivalent to
    circuit_graph that respects couplings in coupling_graph, and
    a layout dict mapping qubits of circuit_graph into qubits
    of coupling_graph. The layout may differ from the initial_layout
    if the first layer of gates cannot be executed on the
    initial_layout.
    """
    if circuit_graph.width() > coupling_graph.size():
        raise QISKitException("Not enough qubits in CouplingGraph")

    # Schedule the input circuit
    layerlist = circuit_graph.layers()
    if verbose:
        print("schedule:")
        for i in range(len(layerlist)):
            print("    %d: %s" % (i, layerlist[i]["partition"]))

    # Check input layout and create default layout if necessary
    if initial_layout is not None:
        circ_qubits = circuit_graph.get_qubits()
        coup_qubits = coupling_graph.get_qubits()
        qubit_subset = []
        for k, v in initial_layout.values():
            qubit_subset.append(v)
            if k not in circ_qubits:
                raise QISKitException("initial_layout qubit %s[%d] not " +
                                      "in input Circuit" % (k[0], k[1]))
            if v not in coup_qubits:
                raise QISKitException("initial_layout qubit %s[%d] not " +
                                      " in input CouplingGraph" % (k[0], k[1]))
    else:
        # Supply a default layout
        qubit_subset = coupling_graph.get_qubits()
        qubit_subset = qubit_subset[0:circuit_graph.width()]
        initial_layout = {a: b for a, b in
                          zip(circuit_graph.get_qubits(), qubit_subset)}

    # Find swap circuit to preceed to each layer of input circuit
    layout = copy.deepcopy(initial_layout)
    openqasm_output = ""
    first_layer = True
    for i in range(len(layerlist)):
        success_flag, best_circ, best_d, best_layout \
          = layer_permutation(layerlist[i]["partition"], layout,
                              qubit_subset, coupling_graph, 20)
        if not success_flag:
            if verbose:
                print("swap_mapper: failed, layer %d, " % i,
                      " contention? retrying sequentially")
            serial_layerlist = layerlist[i]["graph"].serial_layers()
            for j in range(len(serial_layerlist)):
                success_flag, best_circ, best_d, best_layout \
                  = layer_permutation(serial_layerlist[j]["partition"],
                                      layout, qubit_subset, coupling_graph, 20)
                if not success_flag:
                    raise QISKitException("swap_mapper failed: " +
                                          "layer %d, sublayer %d" % (i, j) +
                                          ", \"%s\"" %
                                          serial_layerlist[j]["graph"].qasm(
                                                    no_decls=True,
                                                    aliases=layout))
                else:
                    layout = best_layout
                    if first_layer:
                        initial_layout = layout
                        openqasm_output += circuit_graph.qasm(add_swap=True,
                                                              decls_only=True,
                                                              aliases=layout)
                        openqasm_output += serial_layerlist[j]["graph"].qasm(
                                                        no_decls=True,
                                                        aliases=layout)
                        first_layer = False
                    else:
                        if verbose:
                            print("swap_mapper: layer %d (%d), depth %d"
                                  % (i, j, best_d))
                        if best_circ != "":
                            openqasm_output += best_circ
                        openqasm_output += serial_layerlist[j]["graph"].qasm(
                                                        no_decls=True,
                                                        aliases=layout)
        else:
            layout = best_layout
            if first_layer:
                initial_layout = layout
                openqasm_output += circuit_graph.qasm(add_swap=True,
                                                      decls_only=True,
                                                      aliases=layout)
                openqasm_output += layerlist[i]["graph"].qasm(no_decls=True,
                                                              aliases=layout)
                first_layer = False
            else:
                if verbose:
                    print("swap_mapper: layer %s, depth %d" % (i, best_d))
                if best_circ != "":
                    openqasm_output += best_circ
                openqasm_output += layerlist[i]["graph"].qasm(no_decls=True,
                                                              aliases=layout)
    # Parse openqasm_output into Circuit object
    basis += ",swap"
    ast = Qasm(data=openqasm_output).parse()
    u = unroll.Unroller(ast, unroll.CircuitBackend(basis.split(",")))
    u.execute()
    return u.backend.circuit, initial_layout
