# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""

Description of the algorithm :
Our code decomposes into three main modules:
1. Compiler: Takes a circuit, architecture graph and mapper. Iterates through the circuit layers.
    Calls the mapper to find mappings and mapping circuits.
    Places the mapping circuits.
    Then places as many gates as possible.
    Iterate until done.
2. Mapper: Takes a circuit (layer), an architecture graph and a permuter.
    Tries to place the 2-qubit gates in the circuit such that they can be executed with as little
    overhead as possible.
3. Permuter: Takes an architecture graph and a permutation.
    Computes a short sequence of swaps that implements the permutation on the graph.

- How does the algorithm work?
The core difference between the swap mapper and our code is that our mappers try to compute a
placement of quantum registers onto the architecture _only_. This does not include a circuit.
We then use an algorithm that, given such a mapping, finds a 4-approximation of the best swap
circuit (in size). Our compiler then takes such a mapping, computes the swaps using the routing
algorithm, and finally places the mapping circuit followed by as many gates as possible (without
remapping).

Our starting point for the mapper is the qiskit swap mapper.
We rewrote it to be specifically focused for optimizing in size.
Simplifying it a lot along the way.
We, however, constrain the qiskit mapper quite a bit, by allowing it to try only |V|^2/8 swaps.
When it fails we fall back to our own code, the extension mapper. The extension mapper tries to
first place qubits onto nodes such that the number of swaps is minimized.
Then it tries to extend this placement with more qubits for which it holds that
it is cheaper to place them now than in the next iteration.
Finally, we compute if this procedure has provided a nice placement.
If not, we simply place just the cheapest gate and go to the next iteration.

- Did you use any previously published schemes? Cite relevant papers.
* Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
    ArXiV: https://arxiv.org/abs/1602.05150
    It describes the 4-approximation algorithm. We then generalized it slightly to support partial
    token assignments.

- What packages did you use in your code and for what part of the algorithm?
The packages are listed in the Pipfile(.lock)
qiskit and networkx are the main packages.
Qiskit for the DAGCircuit data structure.
NetworkX for graph operations.

- How general is your approach? Does it work for arbitrary coupling layouts (qubit number)?
Yes it does. If the layout is connected.

- Are there known situations when the algorithm fails?
When running you may encounter https://github.com/QISKit/qiskit-sdk-py/issues/239

Future work:
* Tweak the limit for the qiskit mapper from |V|^2/8 to something better.
* Compute both extension mapper and qiskit mapper circuits and pick the 'best' one.

"""

import re
import copy
import pprint
import logging

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler._basepasses import TransformationPass
from qiskit.mapper import cx_cancellation, optimize_1q_gates

from .src import permutation

from .src.mapping import size
from .src.compiler import compile_to_arch
from .src.mapping.util import direction_mapper
from .src.mapping.placement import Placement
from .src.permutation.general import ApproximateTokenSwapper

logger = logging.getLogger(__name__)


class ExtensionMapper(TransformationPass):
    """Extension mapper implemented as a transpiler pass."""

    def __init__(self, coupling_map):
        """Creates an extension mapper pass instance.

        Args:
           coupling_map (Coupling): Directed graph represented a coupling map.
        """
        self._coupling_map = coupling_map

        super().__init__()

    def run(self, dag):
        circuit = copy.deepcopy(dag)
        circuit.add_basis_element("swap", 2)
        coupling_graph = self._coupling_map.G

        swapper = ApproximateTokenSwapper(coupling_graph.to_undirected(as_view=True))
        size_mapper = size.SizeMapper(coupling_graph,
                                      lambda mapping: swapper.map(mapping, trials=4),
                                      allow_swaps=True)

        def arch_mapper(dag, current_mapping):
            """Uses the mapper to find a mapping of the given dag onto the architecture."""
            partial_mapping = size_mapper.qiskit_mapper(dag, current_mapping)
            logger.debug("Found partial mapping: %s", partial_mapping)
            placement = Placement(current_mapping, partial_mapping)
            # And also the circuit to obtain that mapping.
            swaps, mapping_circuit = placement.mapping_circuit(
                # Optimize the final mapping as much as possible with many trials.
                lambda mapping: ([el] for el in swapper.map(mapping, trials=100)),
                allow_swaps=True)
            logger.debug("The swaps for the partial mapping: %s", swaps)
            inv_mapping = {v: k for k, v in current_mapping.items()}
            permutation.util.swap_permutation(swaps, inv_mapping)
            mapping = {v: k for k, v in inv_mapping.items()}

            return mapping, mapping_circuit

        logger.info("Starting mapping of circuit...")
        compiled_dag, mapping = compile_to_arch(circuit, coupling_graph, arch_mapper)
        logger.info("Done mapping circuit, optimizing...")

        ###
        # Given the DAG that is mapped to the architecture, we now simplify it as much as possible.
        ###
        # Replace swaps by cnots in the right direction
        def remove_swaps(dagcircuit, mapping, coupling_graph):
            """Remove SWAP gates in dagcircuit by replacing with 3 CNOTs.

            The CNOTs are placed such that the number of eventual Hadamard gates is minimised."""
            # Construct a circuit that implements a swap.
            swap_circuit = DAGCircuit()
            swap_circuit.add_basis_element('cx', 2)
            swap_qreg = QuantumRegister(2, name="q")
            swap_circuit.add_qreg(swap_qreg)
            swap_circuit.apply_operation_back("cx", [("q", 0), ("q", 1)])
            swap_circuit.apply_operation_back("cx", [("q", 1), ("q", 0)])
            swap_circuit.apply_operation_back("cx", [("q", 0), ("q", 1)])

            def right_direction(node):
                """Compute if the given two-qubit node has the right orientation"""
                swap_edge = tuple(mapping[qarg] for qarg in node["qargs"])
                if swap_edge in coupling_graph.edges:
                    return True
                if (swap_edge[1], swap_edge[0]) in coupling_graph.edges:
                    return False
                raise RuntimeError("circuit incompatible with CouplingGraph: "
                                   "swap on %s" % pprint.pformat(swap_edge))

            swap_node_list = dagcircuit.get_named_nodes("swap")
            for swap_node in swap_node_list:
                node = dagcircuit.multi_graph.node[swap_node]
                # Find the optimal mapping for the swap_circuit
                if right_direction(node):
                    dagcircuit.substitute_circuit_one(swap_node,
                                                      swap_circuit,
                                                      wires=[("q", 0), ("q", 1)])
                else:
                    dagcircuit.substitute_circuit_one(swap_node,
                                                      swap_circuit,
                                                      wires=[("q", 1), ("q", 0)])
            return len(swap_node_list)

        removed_swaps = remove_swaps(compiled_dag, mapping, coupling_graph)
        # Simplify cx gates
        cx_cancellation(compiled_dag)
        # Change cx directions
        flipped_cnots = direction_mapper(compiled_dag, mapping, coupling_graph)
        logger.debug("Corrected CNOTs: %s", (flipped_cnots - removed_swaps))
        # Simplify single qubit gates
        logger.debug("Running optimize_1q_gates... (This can cause errors)")
        compiled_dag = optimize_1q_gates(compiled_dag)
        logger.debug("Done running optimize_1q_gates.")
        # Rename registers so they are valid identifiers.
        identifier_regex = re.compile('[a-zA-Z][0-9a-zA-Z]*')
        for register in compiled_dag.qregs:
            if not identifier_regex.match(register):
                # IDEA: Use a UUID instead.
                logger.debug("Quantum register %s is not a valid identifier, renaming.", register)
                compiled_dag.rename_register(register, 'mapper' + register)
        for register in compiled_dag.cregs:
            if not identifier_regex.match(register):
                logger.debug("Classical register %s is not a valid identifier, renaming.", register)
                compiled_dag.rename_register(register, 'mapper' + register)

        # nlist = compiled_dag.get_named_nodes("measure")
        # perm = {}
        # for n in nlist:
        #     nd = compiled_dag.multi_graph.node[n]
        #     perm[nd["qargs"][0][1]] = nd["cargs"][0][1]
        # print(perm)
        logger.info("Done with mapping and optimizing.")

        return compiled_dag
