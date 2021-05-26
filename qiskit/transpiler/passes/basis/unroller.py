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

"""Unroll a circuit to a given basis."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlledGate
from qiskit.converters.circuit_to_dag import circuit_to_dag


class Unroller(TransformationPass):
    """Unroll a circuit to a given basis.

    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    def __init__(self, basis):
        """Unroller initializer.

        Args:
            basis (list[str] or None): Target basis names to unroll to, e.g. `['u3', 'cx']` . If
                None, does not unroll any gate.
        """
        super().__init__()
        self.basis = basis

    def run(self, dag):
        """Run the Unroller pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """
        if self.basis is None:
            return dag
        # Walk through the DAG and expand each non-basis node
        basic_insts = ["measure", "reset", "barrier", "snapshot", "delay"]
        for node in dag.op_nodes():
            if node.op._directive:
                continue

            if node.name in basic_insts:
                # TODO: this is legacy behavior.Basis_insts should be removed that these
                #  instructions should be part of the device-reported basis. Currently, no
                #  backend reports "measure", for example.
                continue

            if node.name in self.basis:  # If already a base, ignore.
                if isinstance(node.op, ControlledGate) and node.op._open_ctrl:
                    pass
                else:
                    continue

            # TODO: allow choosing other possible decompositions
            try:
                phase = node.op.definition.global_phase
                rule = node.op.definition.data
            except (TypeError, AttributeError) as err:
                raise QiskitError(
                    f"Error decomposing node of instruction '{node.name}': "
                    f"{err}. Unable to define instruction '{node.name}' in the"
                    f" given basis."
                ) from err

            # Isometry gates definitions can have widths smaller than that of the
            # original gate, in which case substitute_node will raise. Fall back
            # to substitute_node_with_dag if an the width of the definition is
            # different that the width of the node.
            while rule and len(rule) == 1 and len(node.qargs) == len(rule[0][1]) == 1:
                if rule[0][0].name in self.basis:
                    dag.global_phase += phase
                    dag.substitute_node(node, rule[0][0], inplace=True)
                    break
                try:
                    phase += rule[0][0].definition.global_phase
                    rule = rule[0][0].definition.data
                except (TypeError, AttributeError) as err:
                    raise QiskitError(
                        f"Error decomposing node of instruction '{node.name}': "
                        f"{err}. Unable to define instruction '{rule[0][0].name}'"
                        f" in the given basis."
                    ) from err

            else:
                if not rule:
                    if rule == []:  # empty node
                        dag.remove_op_node(node)
                        dag.global_phase += phase
                        continue
                    # opaque node
                    raise QiskitError(
                        "Cannot unroll the circuit to the given basis, %s. "
                        "No rule to expand instruction %s." % (str(self.basis), node.op.name)
                    )
                decomposition = circuit_to_dag(node.op.definition)
                unrolled_dag = self.run(decomposition)  # recursively unroll ops
                dag.substitute_node_with_dag(node, unrolled_dag)
        return dag
