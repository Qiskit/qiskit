# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
DAG Unroller
"""

import networkx as nx

from qiskit._quantumregister import QuantumRegister
from qiskit._classicalregister import ClassicalRegister
from ._unrollererror import UnrollerError
from ._dagbackend import DAGBackend


class DagUnroller(object):
    """An Unroller that takes Dag circuits as the input."""

    def __init__(self, dag_circuit, backend=None):
        if dag_circuit is None:
            raise UnrollerError('Invalid dag circuit!!')

        self.dag_circuit = dag_circuit
        self.set_backend(backend)

    def set_backend(self, backend):
        """Set the backend object.

        Give the same gate definitions to the backend circuit as
        the input circuit.
        """
        self.backend = backend
        for name, data in self.dag_circuit.gates.items():
            self.backend.define_gate(name, data)

    def execute(self):
        """Interpret OPENQASM and make appropriate backend calls.

        This does not expand gates. So self.expand_gates() must have
        been previously called. Otherwise non-basis gates will be ignored
        by this method.
        """
        if self.backend is not None:
            self._process()
            return self.backend.get_output()
        else:
            raise UnrollerError("backend not attached")

    def expand_gates(self, basis=None):
        """Expand all gate nodes to the given basis.

        If basis is empty, each custom gate node is replaced by its
        implementation over U and CX. If basis contains some custom gates,
        then those custom gates are not expanded. For example, if "u3"
        is in basis, then the gate "u3" will not be expanded wherever
        it occurs.

        This method replicates the behavior of the unroller
        module without using the OpenQASM parser or the ast.
        """
        if not basis:
            basis = []
        basis = list(set(self.backend.circuit.basis).union(set(basis)))

        if not isinstance(self.backend, DAGBackend):
            raise UnrollerError("expand_gates only accepts a DAGBackend!!")

        # Walk through the DAG and expand each non-basis node
        simulator_builtins = ['snapshot', 'save', 'load', 'noise']
        topological_sorted_list = list(nx.topological_sort(self.dag_circuit.multi_graph))
        for node in topological_sorted_list:
            current_node = self.dag_circuit.multi_graph.node[node]
            if current_node["type"] == "op" and \
                    current_node["op"].name not in basis and \
                    current_node["op"].name not in simulator_builtins and \
                    not self.dag_circuit.gates[current_node["op"].name]["opaque"]:
                decomposition_rules = current_node["op"]._decompositions
                if not decomposition_rules:
                    raise UnrollerError("no decomposition rules defined for ",
                                        current_node["op"].name)
                # TODO: allow choosing other possible decompositions
                decomposition_dag = decomposition_rules[0]
                condition = current_node["condition"]
                # the decomposition rule must be amended if used in a
                # conditional context. delete the op nodes and replay
                # them with the condition.
                if condition:
                    decomposition_dag.add_creg(condition[0])
                    to_replay = []
                    for n_it in nx.topological_sort(decomposition_dag.multi_graph):
                        n = decomposition_dag.multi_graph.nodes[n_it]
                        if n["type"] == "op":
                            to_replay.append(n)
                    for n in decomposition_dag.get_op_nodes():
                        decomposition_dag._remove_op_node(n)
                    for n in to_replay:
                        decomposition_dag.apply_operation_back(n["op"], condition=condition)

                # the wires for substitute_circuit_one are expected as qargs first,
                # then cargs, then conditions
                qwires = [w for w in decomposition_dag.wires
                          if isinstance(w[0], QuantumRegister)]
                cwires = [w for w in decomposition_dag.wires
                          if isinstance(w[0], ClassicalRegister)]

                self.dag_circuit.substitute_circuit_one(node,
                                                        decomposition_dag,
                                                        qwires + cwires)

        # if still not unrolled down to basis, recurse
        gate_set = set([self.dag_circuit.multi_graph.nodes[n]["op"].name
                        for n in self.dag_circuit.get_op_nodes()])
        if not gate_set.issubset(basis):
            self.expand_gates(basis)

        return self.dag_circuit

    def _process(self):
        """Process dag nodes.

        This method does *not* unroll.
        """
        for qreg in self.dag_circuit.qregs.values():
            self.backend.new_qreg(qreg)
        for creg in self.dag_circuit.cregs.values():
            self.backend.new_creg(creg)
        for n in nx.topological_sort(self.dag_circuit.multi_graph):
            current_node = self.dag_circuit.multi_graph.node[n]
            if current_node["type"] == "op":
                if current_node["condition"] is not None:
                    self.backend.set_condition(current_node["condition"][0],
                                               current_node["condition"][1])

                # TODO: The schema of the snapshot gate is radically
                # different to other QASM instructions. The current model
                # of extensions does not support generating custom Qobj
                # instructions (only custom QASM strings) and the default
                # instruction generator is not enough to produce a valid
                # snapshot instruction for the new Qobj format.
                #
                # This is a hack since there would be mechanisms for the
                # extensions to provide their own Qobj instructions.
                # Extensions should not be hardcoded in the DAGUnroller.
                extra_fields = None
                if current_node["op"].name == "snapshot" and \
                        not isinstance(self.backend, DAGBackend):
                    extra_fields = {'type': 'MISSING', 'label': 'MISSING',
                                    'texparams': []}

                self.backend.start_gate(current_node["op"], qargs=current_node["qargs"],
                                        extra_fields=extra_fields)
                self.backend.end_gate(current_node["op"])

                self.backend.drop_condition()

        return self.backend.get_output()
