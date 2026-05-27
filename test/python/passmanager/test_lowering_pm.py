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


"""Test the lowering pass manager."""

from test import QiskitTestCase
from copy import copy

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.compilation_status import PassManagerState, PropertySet, WorkflowStatus
from qiskit.passmanager.optimization_pm import OptimizationPassManager
from qiskit.passmanager.lowering_pm import LoweringPassManager
from .tasks import CircuitNoOp, CircuitAnalysis, CircuitToDAG, DAGNoOp, DAGRemoveIdentity, RequirePropertySet


class TestLoweringPassManager(QiskitTestCase):
    """Test the lowering pass manager."""

    def test_basic_lowering(self):
        """Test lowering with no pre- or post-tasks produces the correct output."""
        pm = LoweringPassManager(CircuitToDAG())

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)

        result = pm.run(circuit)

        self.assertEqual(circuit_to_dag(circuit), result)

    def test_pre_and_post(self):
        """Test that pre and post lowering returns an OptimizationPassManagers containing the tasks."""
        pre = CircuitNoOp()
        post = DAGNoOp()
        pm = LoweringPassManager(CircuitToDAG(), pre=[pre], post=[post])

        with self.subTest("pre_lowering property"):
            self.assertIsInstance(pm.pre_lowering, OptimizationPassManager)
            self.assertIn(pre, pm.pre_lowering.tasks)

        with self.subTest("post_lowering property"):
            self.assertIsInstance(pm.post_lowering, OptimizationPassManager)
            self.assertIn(post, pm.post_lowering.tasks)

        tasks_called = []

        def callback(*args):
            tasks_called.append(args[0].__class__.__name__)

        _ = pm.run(QuantumCircuit(1), callback=callback)

        with self.subTest("pre and post executed"):
            self.assertEqual(["CircuitNoOp", "CircuitToDAG", "DAGNoOp"], tasks_called)

    def test_callback(self):
        """Test that the callback works and the right IR types are given."""
        pm = LoweringPassManager(
            CircuitToDAG(), pre=[CircuitAnalysis()], post=[DAGRemoveIdentity()]
        )

        records = []

        def callback(task, ir, property_set, running_time, count):
            records.append((task.__class__.__name__, ir, copy(property_set), running_time, count))

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.rz(1e-10, 0)
        pm.run(circuit, callback=callback)

        self.assertEqual(len(records), 3)

        self.assertEqual(records[0][0], "CircuitAnalysis")
        self.assertIsInstance(records[0][1], QuantumCircuit)
        self.assertEqual(records[0][2]["ops"], circuit.count_ops())  # CircuitAnalysis wrote ops

        self.assertEqual(records[1][0], "CircuitToDAG")
        self.assertIsInstance(records[1][1], DAGCircuit)
        self.assertEqual(records[1][2]["ops"], circuit.count_ops())

        self.assertEqual(records[2][0], "DAGRemoveIdentity")
        self.assertIsInstance(records[2][1], DAGCircuit)
        self.assertEqual(records[2][2]["ops"], circuit.count_ops())

        for i, (_, _, _, runtime, count) in enumerate(records):
            self.assertGreaterEqual(runtime, 0.0)
            self.assertEqual(count, i + 1)

    def test_execute(self):
        """Test ``execute`` as entry point."""
        pm = LoweringPassManager(CircuitToDAG(), pre=[CircuitNoOp()])

        state = PassManagerState(property_set=PropertySet(), workflow_status=WorkflowStatus())

        circuit = QuantumCircuit(1)
        circuit.h(0)
        result = pm.execute(circuit, state)

        self.assertEqual(circuit_to_dag(circuit), result)
        self.assertEqual(state.workflow_status.count, 2)

    def test_run(self):
        """Test ``run`` as a standalone entry point."""
        pm = LoweringPassManager(CircuitToDAG(), pre=[CircuitNoOp()])

        counts = []
        circuit = QuantumCircuit(1)
        pm.run(circuit, callback=lambda *args: counts.append(args[-1]))
        pm.run(circuit, callback=lambda *args: counts.append(args[-1]))

        # Each call to run() creates a fresh state, so the count resets to 1 each time
        self.assertEqual(counts, [1, 2, 1, 2])

    def test_run_on_list(self):
        """Test that ``run`` processes a list of programs and returns a list of outputs."""
        pm = LoweringPassManager(CircuitToDAG())

        circuits = [QuantumCircuit(i + 1) for i in range(3)]
        results = pm.run(circuits)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(circuits))
        for circuit, result in zip(circuits, results):
            self.assertIsInstance(result, DAGCircuit)
            self.assertEqual(circuit_to_dag(circuit), result)

    def test_initial_property_set(self):
        """Test that a pre-populated ``property_set`` is visible to tasks during execution."""
        pm = LoweringPassManager(CircuitToDAG(), pre=[RequirePropertySet("seed")])

        initial_ps = PropertySet()
        initial_ps["seed"] = 42

        # Should not raise: the required property is present
        pm.run(QuantumCircuit(1), property_set=initial_ps)

        # Should raise: no property set provided, so "seed" is absent
        with self.assertRaises(ValueError):
            pm.run(QuantumCircuit(1))
