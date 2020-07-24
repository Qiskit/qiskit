# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unrolls instructions with custom definitions."""

from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import ControlledGate
from qiskit.converters.circuit_to_dag import circuit_to_dag


class UnrollCustomDefinitions(TransformationPass):
    """Unrolls instructions with custom definitions."""

    def __init__(self, equivalence_library, basis_gates):
        """Unrolls instructions with custom definitions.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            basis_gates (list[str]): Target basis names to unroll to, e.g. `['u3', 'cx']`.
        """

        super().__init__()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates

    def run(self, dag):
        """Run the UnrollCustomDefinitions pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """

        if self._basis_gates is None:
            return dag

        basic_insts = set(('measure', 'reset', 'barrier', 'snapshot'))
        device_insts = basic_insts | set(self._basis_gates)

        for node in dag.op_nodes():

            if node.name in device_insts or self._equiv_lib.has_entry(node.op):
                if isinstance(node.op, ControlledGate) and node.op._open_ctrl:
                    pass
                else:
                    continue

            try:
                rule = node.op.definition.data
            except TypeError as err:
                raise QiskitError('Error decomposing node {}: {}'.format(node.name, err))
            except AttributeError:
                # definition is None
                rule = None

            if not rule:
                if rule == []:
                    dag.remove_op_node(node)
                    continue

                # opaque node
                raise QiskitError("Cannot unroll the circuit to the given basis, %s. "
                                  "Instruction %s not found in equivalence library "
                                  "and no rule found to expand." %
                                  (str(self._basis_gates), node.op.name))
            decomposition = circuit_to_dag(node.op.definition)
            unrolled_dag = UnrollCustomDefinitions(self._equiv_lib,
                                                   self._basis_gates).run(
                                                       decomposition)
            dag.substitute_node_with_dag(node, unrolled_dag)

        return dag
