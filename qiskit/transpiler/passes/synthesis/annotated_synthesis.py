# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Default plugin for synthesis of annotated operations.
"""

from __future__ import annotations

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.operation import Operation
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    Modifier,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.transpiler.exceptions import TranspilerError

from qiskit._accelerate.high_level_synthesis import py_synthesize_operation
from .plugin import HighLevelSynthesisPlugin


class AnnotatedSynthesisDefault(HighLevelSynthesisPlugin):
    """Synthesize an :class:`.AnnotatedOperation` using the default synthesis algorithm.

    This plugin name is:``annotated.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, AnnotatedOperation):
            return None

        operation = high_level_object
        modifiers = high_level_object.modifiers

        # The plugin needs additional information that is not yet passed via the run's method
        # arguments: namely high-level-synthesis data and options, the global qubits over which
        # the operation is defined, and the initial state of each global qubit.
        tracker = options.get("qubit_tracker", None)
        data = options.get("hls_data", None)
        input_qubits = options.get("input_qubits", None)

        if data is None or input_qubits is None:
            raise TranspilerError(
                "HighLevelSynthesis: problem with the default plugin for annotated operations."
            )

        if len(modifiers) > 0:
            num_ctrl = sum(
                mod.num_ctrl_qubits for mod in modifiers if isinstance(mod, ControlModifier)
            )
            total_power = sum(mod.power for mod in modifiers if isinstance(mod, PowerModifier))
            is_inverted = sum(1 for mod in modifiers if isinstance(mod, InverseModifier)) % 2

            # The base operation cannot use control qubits as auxiliary qubits.
            # In addition, when we have power or inverse modifiers, we need to set all of
            # the operation's qubits to dirty. Note that synthesizing the base operation we
            # can use additional auxiliary qubits, however they would always be returned to
            # their previous state, so clean qubits remain clean after each for- or while- loop.
            annotated_tracker = tracker.copy()
            annotated_tracker.disable(input_qubits[:num_ctrl])  # do not access control qubits
            if total_power != 0 or is_inverted:
                annotated_tracker.set_dirty(input_qubits)

            # First, synthesize the base operation of this annotated operation.
            # Note that synthesize_operation also returns the output qubits on which the
            # operation is defined, however currently the plugin mechanism has no way
            # to return these (and instead the upstream code greedily grabs some ancilla
            # qubits from the circuit). We should refactor the plugin "run" iterface to
            # return the actual ancilla qubits used.
            synthesized_base_op_result = py_synthesize_operation(
                operation.base_op, input_qubits[num_ctrl:], data, annotated_tracker
            )

            # The base operation does not need to be synthesized.
            # For simplicity, we wrap the instruction into a circuit. Note that
            # this should not deteriorate the quality of the result.
            if synthesized_base_op_result is None:
                synthesized_base_op = _instruction_to_circuit(operation.base_op)
            else:
                synthesized_base_op = QuantumCircuit._from_circuit_data(
                    synthesized_base_op_result[0]
                )
            tracker.set_dirty(input_qubits[num_ctrl:])

            # This step currently does not introduce ancilla qubits. However it makes
            # a lot of sense to allow this in the future.
            synthesized = _apply_annotations(synthesized_base_op, operation.modifiers)

            if not isinstance(synthesized, QuantumCircuit):
                raise TranspilerError(
                    "HighLevelSynthesis: problem with the default plugin for annotated operations."
                )

            return synthesized

        return None


def _apply_annotations(circuit: QuantumCircuit, modifiers: list[Modifier]) -> QuantumCircuit:
    """
    Applies modifiers to a quantum circuit.
    """

    if not isinstance(circuit, QuantumCircuit):
        raise TranspilerError("HighLevelSynthesis: incorrect input to 'apply_annotations'.")

    for modifier in modifiers:
        if isinstance(modifier, InverseModifier):
            circuit = circuit.inverse()

        elif isinstance(modifier, ControlModifier):
            if circuit.num_clbits > 0:
                raise TranspilerError(
                    "HighLevelSynthesis: cannot control a circuit with classical bits."
                )

            # Apply the control modifier to each gate in the circuit.
            controlled_circuit = QuantumCircuit(modifier.num_ctrl_qubits + circuit.num_qubits)
            if circuit.global_phase != 0:
                controlled_op = GlobalPhaseGate(circuit.global_phase).control(
                    num_ctrl_qubits=modifier.num_ctrl_qubits,
                    label=None,
                    ctrl_state=modifier.ctrl_state,
                    annotated=False,
                )
                controlled_qubits = list(range(0, modifier.num_ctrl_qubits))
                controlled_circuit.append(controlled_op, controlled_qubits)
            for inst in circuit:
                inst_op = inst.operation
                inst_qubits = inst.qubits
                controlled_op = inst_op.control(
                    num_ctrl_qubits=modifier.num_ctrl_qubits,
                    label=None,
                    ctrl_state=modifier.ctrl_state,
                    annotated=False,
                )
                controlled_qubits = list(range(0, modifier.num_ctrl_qubits)) + [
                    modifier.num_ctrl_qubits + circuit.find_bit(q).index for q in inst_qubits
                ]
                controlled_circuit.append(controlled_op, controlled_qubits)

            circuit = controlled_circuit

            if isinstance(circuit, AnnotatedOperation):
                raise TranspilerError(
                    "HighLevelSynthesis: failed to synthesize the control modifier."
                )

        elif isinstance(modifier, PowerModifier):
            circuit = circuit.power(modifier.power)

        else:
            raise TranspilerError(f"HighLevelSynthesis: Unknown modifier {modifier}.")

    if not isinstance(circuit, QuantumCircuit):
        raise TranspilerError("HighLevelSynthesis: incorrect output of 'apply_annotations'.")

    return circuit


def _instruction_to_circuit(op: Operation) -> QuantumCircuit:
    """Wraps a single operation into a quantum circuit."""
    circuit = QuantumCircuit(op.num_qubits, op.num_clbits)
    circuit.append(op, circuit.qubits, circuit.clbits)
    return circuit
