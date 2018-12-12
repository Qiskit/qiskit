# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
DAG Unroller
"""

import networkx as nx

from ._unrollererror import UnrollerError


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

        This does not expand gates. So Unroller must have
        been previously called. Otherwise non-basis gates will be ignored
        by this method.
        """
        if self.backend is not None:
            self._process()
            return self.backend.get_output()
        else:
            raise UnrollerError("backend not attached")

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
                if current_node["op"].name == "snapshot":
                    extra_fields = {'type': 'MISSING', 'label': 'MISSING',
                                    'texparams': []}

                self.backend.start_gate(current_node["op"],
                                        qargs=current_node["qargs"],
                                        cargs=current_node["cargs"],
                                        extra_fields=extra_fields)
                self.backend.end_gate(current_node["op"])

                self.backend.drop_condition()

        return self.backend.get_output()
