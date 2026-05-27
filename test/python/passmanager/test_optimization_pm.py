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


"""Test the optimization pass manager."""

from test import QiskitTestCase
from copy import copy
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.passmanager.compilation_status import PassManagerState, PropertySet, WorkflowStatus
from qiskit.passmanager.exceptions import PassManagerError
from qiskit.passmanager.optimization_pm import OptimizationPassManager
from qiskit.transpiler import generate_preset_clifford_t_pass_manager
import qiskit.transpiler.passes as dag_passes
from .tasks import (
    CircuitNoOp,
    CircuitAnalysis,
    CircuitRemoveBarriers,
    CircuitRemoveIdentity,
    DAGRemoveBarriers,
    RequireKey,
)


class TestOptimizationPassManager(QiskitTestCase):
    """Test the optimization pass manager."""

    def setUp(self):
        super().setUp()
        self.circuit_pm = OptimizationPassManager(
            [CircuitNoOp(), CircuitRemoveBarriers(), CircuitRemoveIdentity(), CircuitAnalysis()]
        )

    def test_no_tasks(self):
        """Test that empty task lists."""
        with self.subTest("None"):
            self.assertEqual([], OptimizationPassManager(None).tasks)

        with self.subTest("empty list"):
            self.assertEqual([], OptimizationPassManager([]).tasks)

        with self.subTest("empty iter"):
            self.assertEqual([], OptimizationPassManager(iter([])).tasks)

    def test_run(self):
        """Test running an optimization PM."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.barrier()
        circuit.id(0)
        circuit.barrier()
        circuit.z(0)

        out = self.circuit_pm.run(circuit)
        expected = QuantumCircuit(1)
        expected.x(0)
        expected.z(0)

        self.assertEqual(expected, out)

    def test_run_iterable(self):
        """Test that run processes a list of programs and returns a list of outputs."""
        circuit1 = QuantumCircuit(1)
        circuit1.x(0)
        circuit1.barrier()
        circuit1.id(0)
        circuit1.barrier()
        circuit1.z(0)

        circuit2 = QuantumCircuit(2, 1)
        circuit2.cx(0, 1)
        circuit2.barrier()
        circuit2.measure(1, 0)

        expected1 = QuantumCircuit(1)
        expected1.x(0)
        expected1.z(0)

        expected2 = QuantumCircuit(2, 1)
        expected2.cx(0, 1)
        expected2.measure(1, 0)

        with self.subTest(msg="list"):
            out = self.circuit_pm.run([circuit1, circuit2])
            self.assertEqual(expected1, out[0])
            self.assertEqual(expected2, out[1])

        with self.subTest(msg="generator"):
            out = self.circuit_pm.run(iter((circuit1, circuit2)))
            self.assertEqual(expected1, out[0])
            self.assertEqual(expected2, out[1])

    def test_run_initial_property_set(self):
        """Test that a pre-populated property_set passed to run is visible to tasks."""
        pm = OptimizationPassManager([RequireKey("clean_socks")])

        circuit = QuantumCircuit()
        with self.subTest("valid property set"):
            out = pm.run(circuit, property_set={"clean_socks": True})
            self.assertEqual(circuit, out)

        with self.subTest("missing property"):
            with self.assertRaises(ValueError):
                _ = pm.run(circuit)

    def test_execute(self):
        """Test the execute interface."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.barrier()
        circuit.id(0)
        circuit.barrier()
        circuit.z(0)

        out = self.circuit_pm.run(circuit)
        expected = QuantumCircuit(1)
        expected.x(0)
        expected.z(0)

        state = PassManagerState(property_set=PropertySet(), workflow_status=WorkflowStatus())
        out, state = self.circuit_pm.execute(circuit, state)

        self.assertEqual(expected, out)
        self.assertEqual(4, state.workflow_status.count)
        self.assertEqual(2, state.property_set["removed_barriers"])

    def test_tasks(self):
        """Test that the tasks property returns the current task list."""
        tasks = [CircuitNoOp(), CircuitNoOp()]
        pm = OptimizationPassManager(tasks)
        self.assertEqual(pm.tasks, tasks)

    def test_append_tasks(self):
        """Test appending tasks."""
        pm = OptimizationPassManager([])

        noop = CircuitNoOp()
        pm.append(noop)
        self.assertEqual(pm.tasks, [noop])

        pm.append([noop, noop, noop])
        self.assertEqual(pm.tasks, [noop, noop, noop, noop])

    def test_append_invalid_type(self):
        """Test that appending a non-Task object raises TypeError."""
        pm = OptimizationPassManager([])
        with self.assertRaises(TypeError):
            pm.append("not tonight, mate")

    def test_replace(self):
        """Test that replace substitutes the task at the given index."""
        noop = CircuitNoOp()
        barriers = CircuitRemoveBarriers()

        pm = OptimizationPassManager([barriers])
        pm.replace(0, noop)
        self.assertEqual(noop, pm.tasks[0])

    def test_replace_out_of_bounds(self):
        """Test that replace with an out-of-bounds index raises PassManagerError."""
        pm = OptimizationPassManager([])
        with self.assertRaises(PassManagerError):
            pm.replace(42, CircuitNoOp())

    def test_remove(self):
        """Test removing."""
        noop = CircuitNoOp()
        barriers = CircuitRemoveBarriers()

        pm = OptimizationPassManager([barriers])
        pm.replace(0, noop)
        self.assertEqual(noop, pm.tasks[0])

    def test_remove_out_of_bounds(self):
        """Test removing out of bounds fails."""
        pm = OptimizationPassManager([])
        with self.assertRaises(PassManagerError):
            pm.remove(0)

    def test_callback(self):
        """Test the callback."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.barrier()
        circuit.id(0)
        circuit.barrier()
        circuit.z(0)

        no_barriers = QuantumCircuit(1)
        no_barriers.x(0)
        no_barriers.id(0)
        no_barriers.z(0)

        final = QuantumCircuit(1)
        final.x(0)
        final.z(0)

        records = []

        def callback(task, passmanager_ir, property_set, running_time, count):
            records.append(
                (
                    task.__class__.__name__,
                    copy(passmanager_ir),
                    copy(property_set),
                    running_time,
                    count,
                )
            )

        out = self.circuit_pm.run(circuit, callback=callback)

        expected_names = [
            "CircuitNoOp",
            "CircuitRemoveBarriers",
            "CircuitRemoveIdentity",
            "CircuitAnalysis",
        ]
        expected_programs = [circuit, no_barriers, final, final]
        expected_property_sets = [
            {},
            {"removed_barriers": 2},
            {"removed_barriers": 2},
            {"removed_barriers": 2, "ops": final.count_ops()},
        ]

        for i, (name, program, props, time, count) in enumerate(records):
            with self.subTest(i=i):
                self.assertEqual(name, expected_names[i])
                self.assertEqual(program, expected_programs[i])
                self.assertEqual(props, expected_property_sets[i])
                self.assertGreaterEqual(time, 0.0)
                self.assertEqual(count, i + 1)

        self.assertEqual(out, final)

    def test_nested_as_task(self):
        """Test that an OptimizationPassManager can itself be used as a task inside another one."""
        inner = OptimizationPassManager([CircuitNoOp()] * 10)
        outer = OptimizationPassManager([CircuitNoOp(), inner])

        state = PassManagerState(workflow_status=WorkflowStatus(), property_set=PropertySet())
        _, state = outer.execute(QuantumCircuit(), state)

        self.assertEqual(state.workflow_status.count, 11)

    def test_conditional(self):
        """Test a conditional controller."""

        def if_has_barrier(property_set: PropertySet):
            if (ops := property_set.get("ops")) is None:
                raise RuntimeError("Required `ops` property is missing")
            return "barrier" in ops

        pm = OptimizationPassManager(
            [CircuitAnalysis(), ConditionalController(CircuitRemoveBarriers(), if_has_barrier)]
        )

        circuit1 = QuantumCircuit(10)
        circuit1.barrier()
        expected_passes1 = {pm.tasks[0], pm.tasks[1].tasks[0]}

        circuit2 = QuantumCircuit(10)
        circuit2.x(range(10))
        expected_passes2 = {pm.tasks[0]}

        for circuit, expected_passes in zip(
            [circuit1, circuit2], [expected_passes1, expected_passes2]
        ):
            with self.subTest(circuit=circuit):
                state = PassManagerState(
                    workflow_status=WorkflowStatus(), property_set=PropertySet()
                )
                _, state = pm.execute(circuit, state)

                self.assertEqual(expected_passes, state.workflow_status.completed_passes)

    def test_dag_optimization(self):
        """Test compatibility with existing passes on the DAG."""
        pm = OptimizationPassManager(
            [
                dag_passes.Collect2qBlocks(),
                dag_passes.ConsolidateBlocks(),
                dag_passes.Split2QUnitaries(),
                dag_passes.Optimize1qGatesDecomposition(basis=["u", "cx"]),
                dag_passes.InverseCancellation(),
            ]
        )

        circuit = QuantumCircuit(2)
        circuit.rz(2.6, 1)
        circuit.rxx(np.pi, 0, 1)
        circuit.ry(0.23, 0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)

        dag = circuit_to_dag(circuit)
        out = pm.run(dag)

        self.assertEqual(out.count_ops(), {"u": 2})

    def test_generate_preset_as_task(self):
        """Test using a generate preset PM as task."""
        pm = OptimizationPassManager(
            [
                DAGRemoveBarriers(),
                generate_preset_clifford_t_pass_manager(basis_gates=["t", "h", "cx"]),
            ]
        )

        circuit = QuantumCircuit(2)
        circuit.rz(0.25, 0)
        circuit.barrier()
        circuit.crx(0.5, 0, 1)
        circuit.h(0)
        circuit.barrier()
        circuit.h(0)

        state = PassManagerState(workflow_status=WorkflowStatus(), property_set=PropertySet())
        out, state = pm.execute(circuit_to_dag(circuit), state)

        self.assertEqual(out.count_ops().keys(), {"t", "h", "cx"})
        self.assertEqual(2, state.property_set["removed_barriers"])
        # we run DAGRemoveBarriers, plus all passes inside generate_preset_clifford_t
        self.assertGreater(state.workflow_status.count, 2)
