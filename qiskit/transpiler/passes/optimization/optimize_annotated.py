# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimize annotated operations on a circuit."""

from typing import Optional, Union, List, Tuple

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.annotated_operation import AnnotatedOperation, _canonicalize_modifiers
from qiskit.circuit import EquivalenceLibrary, ControlledGate, Operation
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError


class OptimizeAnnotated(TransformationPass):
    """Optimization pass on circuits with annotated operations."""

    def __init__(
        self,
        target: Optional[Target] = None,
        equivalence_library: Optional[EquivalenceLibrary] = None,
        basis_gates: Optional[List[str]] = None,
    ):
        """
        OptimizeAnnotated initializer.

        Args:
            target: Optional, the backend target to use for this pass.
            equivalence_library: The equivalence library used
                (instructions in this library will not be optimized by this pass).
            basis_gates: Optional, target basis names to unroll to, e.g. `['u3', 'cx']`
                (instructions in this list will not be optimized by this pass).
                Ignored if ``target`` is also specified.
        """
        super().__init__()

        self._target = target
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates

        self._top_level_only = self._basis_gates is None and self._target is None

        if not self._top_level_only and self._target is None:
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay"}
            self._device_insts = basic_insts | set(self._basis_gates)

    def run(self, dag: DAGCircuit):
        """Run the OptimizeAnnotated pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations optimized.

        Raises:
            TranspilerError: when something goes wrong.

        """
        dag, _ = self._run_inner(dag)
        return dag

    def _run_inner(self, dag) -> Tuple[DAGCircuit, bool]:
        """
        Optimizes annotated operations.
        Returns True if did something.
        """
        # First, optimize every node in the DAG.
        dag, opt1 = self._canonicalize(dag)

        opt2 = False
        if not self._top_level_only:
            # Second, recursively descend into definitions.
            # Note that it is important to recurse only after the optimization methods have been run,
            # as they may remove annotated gates.
            dag, opt2 = self._recurse(dag)

        return dag, opt1 or opt2

    def _canonicalize(self, dag) -> Tuple[DAGCircuit, bool]:
        """
        Combines recursive annotated operations and canonicalizes modifiers.
        Returns True if did something.
        """

        did_something = False
        for node in dag.op_nodes():
            if isinstance(node.op, AnnotatedOperation):
                modifiers = []
                cur = node.op
                while isinstance(cur, AnnotatedOperation):
                    modifiers.extend(cur.modifiers)
                    cur = cur.base_op
                canonical_modifiers = _canonicalize_modifiers(modifiers)
                if len(canonical_modifiers) > 0:
                    # this is still an annotated operation
                    node.op.base_op = cur
                    node.op.modifiers = canonical_modifiers
                else:
                    # no need for annotated operations
                    node.op = cur
                did_something = True
        return dag, did_something

    def _recursively_process_definitions(self, op: Operation) -> bool:
        """
        Recursively applies optimizations to op's definition (or to op.base_op's
        definition if op is an annotated operation).
        Returns True if did something.
        """

        # If op is an annotated operation, we descend into its base_op
        if isinstance(op, AnnotatedOperation):
            return self._recursively_process_definitions(op.base_op)

        # Similar to HighLevelSynthesis transpiler pass, we do not recurse into a gate's
        # `definition` for a gate that is supported by the target or in equivalence library.

        controlled_gate_open_ctrl = isinstance(op, ControlledGate) and op._open_ctrl
        if not controlled_gate_open_ctrl:
            inst_supported = (
                self._target.instruction_supported(operation_name=op.name)
                if self._target is not None
                else op.name in self._device_insts
            )
            if inst_supported or (self._equiv_lib is not None and self._equiv_lib.has_entry(op)):
                return False

        try:
            # extract definition
            definition = op.definition
        except TypeError as err:
            raise TranspilerError(
                f"OptimizeAnnotated was unable to extract definition for {node.op.name}: {err}"
            ) from err
        except AttributeError:
            # definition is None
            definition = None

        if definition is None:
            raise TranspilerError(f"OptimizeAnnotated was unable to optimize {node.op}.")

        definition_dag = circuit_to_dag(definition, copy_operations=False)
        definition_dag, opt = self._run_inner(definition_dag)

        if opt:
            # We only update a gate's definition if it was actually changed.
            # This is important to preserve non-annotated singleton gates.
            op.definition = dag_to_circuit(definition_dag)

        return opt

    def _recurse(self, dag) -> Tuple[DAGCircuit, bool]:
        """
        Recursively handles gate definitions.
        Returns True if did something.
        """

        did_something = False

        # Don't do anything else if processing only top-level
        if self._top_level_only:
            return dag, did_something

        for node in dag.op_nodes():
            # Similar to HighLevelSynthesis transpiler pass, we do not recurse into a gate's
            # `definition` for a gate that is supported by the target or in equivalence library.
            opt = self._recursively_process_definitions(node.op)
            did_something = did_something or opt

        return dag, did_something
