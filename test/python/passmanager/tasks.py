# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Shared tasks for pass manager tests."""

import time
from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit, Barrier
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.compilation_status import PassManagerState, PropertySet
from qiskit.passmanager.base_tasks import Task, IR, IR_OUT, Callback
from qiskit.transpiler.passes import RemoveIdentityEquivalent


class BaseTask(Task[IR, IR_OUT], ABC):
    """A simple base class for tasks, which implements callback info and input type checking."""

    def __init__(self, in_type: type):
        self.in_type = in_type

    def execute(
        self, passmanager_ir: IR, state: PassManagerState, callback: Callback[IR_OUT] | None = None
    ) -> tuple[IR_OUT, PassManagerState]:
        if not isinstance(passmanager_ir, self.in_type):
            raise TypeError(f"expected {self.in_type.__name__}")

        start = time.time()
        passmanager_ir = self.run(passmanager_ir, state.property_set)
        runtime = time.time() - start

        state.workflow_status.count += 1
        state.workflow_status.completed_passes.add(self)
        if callback is not None:
            callback(
                task=self,
                passmanager_ir=passmanager_ir,
                property_set=state.property_set,
                running_time=runtime,
                count=state.workflow_status.count,
            )

        return passmanager_ir, state

    @abstractmethod
    def run(self, passmanager_ir: IR, property_set: PropertySet | None = None) -> IR_OUT: ...


class CircuitNoOp(BaseTask[QuantumCircuit, QuantumCircuit]):
    """A dummy task preserving ``QuantumCircuit`` as IR."""

    def __init__(self):
        super().__init__(QuantumCircuit)

    def run(self, passmanager_ir, property_set):
        return passmanager_ir


class CircuitAnalysis(BaseTask[QuantumCircuit, QuantumCircuit]):
    """A task counting the number of operations and storing them in the property set."""

    def __init__(self):
        super().__init__(QuantumCircuit)

    def run(self, passmanager_ir, property_set):
        property_set["ops"] = passmanager_ir.count_ops()
        return passmanager_ir


class CircuitToDAG(BaseTask[QuantumCircuit, DAGCircuit]):
    """A lowering task from circuit to DAG."""

    def __init__(self):
        super().__init__(QuantumCircuit)

    def run(self, passmanager_ir, property_set):
        return circuit_to_dag(passmanager_ir)


class CircuitRemoveBarriers(BaseTask[QuantumCircuit, QuantumCircuit]):
    """Remove all barriers."""

    def __init__(self):
        super().__init__(QuantumCircuit)

    def run(self, passmanager_ir, property_set):
        out = passmanager_ir.copy_empty_like()
        count = 0

        for inst in passmanager_ir.data:
            if isinstance(inst.operation, Barrier):
                count += 1
            else:
                out.append(inst.operation, inst.qubits, inst.clbits)

        property_set["removed_barriers"] = count
        return out


class CircuitRemoveIdentity(BaseTask[QuantumCircuit, QuantumCircuit]):
    """Remove all close-to-identity gates."""

    def __init__(self):
        super().__init__(QuantumCircuit)

    def run(self, passmanager_ir, property_set):
        return RemoveIdentityEquivalent()(passmanager_ir)


class DAGRemoveBarriers(BaseTask[DAGCircuit, DAGCircuit]):
    """Remove all barriers."""

    def __init__(self):
        super().__init__(DAGCircuit)

    def run(self, passmanager_ir, property_set):
        out = passmanager_ir.copy_empty_like()
        count = 0

        for op_node in passmanager_ir.topological_op_nodes():
            if isinstance(op_node.op, Barrier):
                count += 1
            else:
                out.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

        property_set["removed_barriers"] = count
        return out


class DAGNoOp(BaseTask[DAGCircuit, DAGCircuit]):
    """A dummy task preserving ``DAGCircuit`` as IR."""

    def __init__(self):
        super().__init__(DAGCircuit)

    def run(self, passmanager_ir, property_set):
        return passmanager_ir


class DAGRemoveIdentity(BaseTask[DAGCircuit, DAGCircuit]):
    """A task removing (close to) identity gates on a DAG."""

    def __init__(self):
        super().__init__(DAGCircuit)
        self._pass = RemoveIdentityEquivalent()

    def run(self, passmanager_ir, property_set):
        return self._pass.run(passmanager_ir)


class RecordOrder(BaseTask[QuantumCircuit, QuantumCircuit]):
    """A task that appends a label to a shared list to record execution order."""

    def __init__(self, log: list, label: str):
        super().__init__(QuantumCircuit)
        self.log = log
        self.label = label

    def run(self, passmanager_ir, property_set):
        self.log.append(self.label)
        return passmanager_ir


class RequireKey(BaseTask[QuantumCircuit, QuantumCircuit]):
    """A task that raises if a required key is absent from the property set."""

    def __init__(self, key: str):
        super().__init__(QuantumCircuit)
        self.key = key

    def run(self, passmanager_ir, property_set):
        if self.key not in property_set:
            raise ValueError(f"Required property ({self.key}) is not set.")
        return passmanager_ir
