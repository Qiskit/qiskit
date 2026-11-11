# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import (
    Task,
    PassContextHandle,
    PassManagerState,
    Pass,
    WorkflowStatus,
    PropertySet,
)


class CircuitToDag(Pass[QuantumCircuit, DAGCircuit]):
    _qiskit_pass_ir_in_ = QuantumCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def _qiskit_pass_run_(self, ir, context):
        return ir.to_dag(copy_operations=False)


class DagToCircuit(Pass[DAGCircuit, QuantumCircuit]):
    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = QuantumCircuit

    def _qiskit_pass_run_(self, ir, context):
        return ir.to_circuit(copy_operations=False)


class LegacyDagPass(Pass[DAGCircuit]):
    """A wrapper for a legacy Python pass."""

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    # the key in the pass context under which we store the written keys in the context
    # -- this is used to convert from/to property sets, since the PassContextHandle does
    # not have a way to iterate over the (Python) keys
    _legacy_keys = "___legacy_keys"

    def __init__(self, legacy_task: Task[DAGCircuit, DAGCircuit]):
        super().__init__()
        self._task = legacy_task

    def _qiskit_pass_run_(self, ir, context):
        state = self._state_from_context(context)
        out, out_state = self._task.execute(ir, state, None)
        self._update_context(context, state, out_state)

        return out

    @staticmethod
    def _state_from_context(context: PassContextHandle) -> PassManagerState:
        dummy_status = WorkflowStatus()
        property_set = PropertySet(
            {key: context[key] for key in context.get(LegacyDagPass._legacy_keys, ())}
        )
        return PassManagerState(dummy_status, property_set)

    @staticmethod
    def _update_context(
        context: PassContextHandle, old_state: PassManagerState, new_state: PassManagerState
    ):
        context[LegacyDagPass._legacy_keys] = []
        for key, value in new_state.property_set.items():
            context[LegacyDagPass._legacy_keys].append(key)
            context[key] = value

        for deleted_key in set(old_state.property_set.keys()).difference(
            new_state.property_set.keys()
        ):
            del context[deleted_key]
