# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if a DAG circuit is already mapped to a coupling map."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.target import Target
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.converters import circuit_to_dag


class CheckMap(AnalysisPass):
    """Check if a DAG circuit is already mapped to a coupling map.

    Check if a DAGCircuit is mapped to ``coupling_map`` by checking that all
    2-qubit interactions are laid out to be on adjacent qubits in the global coupling
    map of the device, setting the property set field (either specified with ``property_set_field``
    or the default ``is_swap_mapped``) to ``True`` or ``False`` accordingly. Note this does not
    validate directionality of the connectivity between  qubits. If you need to check gates are
    implemented in a native direction for a target use the :class:`~.CheckGateDirection` pass
    instead.
    """

    def __init__(self, coupling_map, property_set_field=None):
        """CheckMap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): Directed graph representing a coupling map.
            property_set_field (str): An optional string to specify the property set field to
                store the result of the check. If not default the result is stored in
                ``"is_swap_mapped"``.
        """
        super().__init__()
        if property_set_field is None:
            self.property_set_field = "is_swap_mapped"
        else:
            self.property_set_field = property_set_field
        if isinstance(coupling_map, Target):
            cmap = coupling_map.build_coupling_map()
        else:
            cmap = coupling_map
        if cmap is None:
            self.qargs = None
        else:
            self.qargs = set()
            for edge in cmap.get_edges():
                self.qargs.add(edge)
                self.qargs.add((edge[1], edge[0]))

    def run(self, dag):
        """Run the CheckMap pass on `dag`.

        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        """
        if not self.qargs:
            self.property_set[self.property_set_field] = True
            return
        wire_map = {bit: index for index, bit in enumerate(dag.qubits)}
        self.property_set[self.property_set_field] = self._recurse(dag, wire_map)

    def _recurse(self, dag, wire_map) -> bool:
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for inner, outer in zip(block.qubits, node.qargs)
                    }
                    if not self._recurse(circuit_to_dag(block), inner_wire_map):
                        return False
            elif (
                len(node.qargs) == 2
                and not dag.has_calibration_for(node)
                and (wire_map[node.qargs[0]], wire_map[node.qargs[1]]) not in self.qargs
            ):
                self.property_set["check_map_msg"] = (
                    f"{node.name}({wire_map[node.qargs[0]]}, {wire_map[node.qargs[1]]}) failed"
                )
                return False
        return True
