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


"""Pass manager test cases."""

from test import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import Pass, PassManager, CallbackType, Callback
from qiskit.transpiler import Target, passes, generate_preset_pass_manager


class SetLayout(Pass):
    def __init__(self, target: Target):
        self.target = target
        self.properties = None

    def run(self, ir, context):
        if ir.num_qubits > self.target.num_qubits:
            raise ValueError("Not enough qubits")

        context.set("layout", list(range(ir.num_qubits)))
        self.properties = context
        return ir


class ResetComputationalQubits(Pass):
    def run(self, ir, context):
        if (layout := context.get("layout")) is None:
            raise RuntimeError("'layout' not available")

        ir.reset(layout)
        return ir


class CallbackTester(Callback):
    def __init__(self, hookpoint, required_keys={}):
        self.hookpoint = hookpoint
        self.counter = 0
        self.required_keys = required_keys

    def trigger(self, hookpoint):
        return hookpoint == self.hookpoint

    def ir_and_context(self, ir, context):
        self.counter += 1
        for key in self.required_keys:
            if context.get(key, None) is None:
                raise ValueError("Missing key: %s", key)


class CallbackPrinter(Callback):
    def __init__(self):
        self.pass_names = set()

    def trigger(self, hookpoint):
        return hookpoint == CallbackType.PostPass

    def with_pass(self, pass_, ir, context):
        self.pass_names.add(pass_.__class__.__name__)


class CircuitToDAG(Pass):
    def run(self, ir, context):
        return ir.to_dag(copy_operations=False)


class TestPassManager(QiskitTestCase):
    """Pass manager tests."""

    def test_pass(self):
        """Test a simple pass setup."""

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.t(1)
        circuit.cx(0, 1)

        target = Target(num_qubits=10)

        pm = PassManager()
        pm.push(SetLayout(target))
        pm.push(ResetComputationalQubits())

        out, context = pm.run(circuit)

        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual(out.count_ops().get("reset", 0), 2)
        self.assertEqual(context.get("layout", []), [0, 1])

    def test_callback(self):
        """Test the callbacks."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.t(1)
        circuit.cx(0, 1)

        target = Target(num_qubits=10)

        pm = PassManager()
        pm.push(SetLayout(target))
        pm.push(ResetComputationalQubits())

        for hookpoint, expected_count in [(CallbackType.PostPass, 2)]:
            with self.subTest(hookpoint=hookpoint):
                callback = CallbackTester(hookpoint, required_keys={"layout"})
                _, _ = pm.run(circuit, callback)
                self.assertEqual(callback.counter, expected_count)

    def test_unsupported_pass_type(self):
        """Test an unsupported pass type raises."""
        with self.assertRaisesRegex(TypeError, "Unsupported pass type"):
            PassManager().push("potato salad")

    def test_legacy_passes(self):
        """Test calling single legacy passes."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.rz(0.2, 0)
        circuit.cx(0, 1)

        pm = PassManager()
        pm.push(CircuitToDAG())
        pm.push(passes.SynthesizeRZRotations())

        out, _ = pm.run(circuit)

        self.assertIsInstance(out, DAGCircuit)
        self.assertTrue("rz" not in out.count_ops().keys())

    def test_legacy_pipeline(self):
        """Test a full compiler pipeline from earlier."""
        full_pm = generate_preset_pass_manager()
        pm = PassManager()
        pm.push(CircuitToDAG())

        all_passes = {}
        for i, task in enumerate(full_pm.to_flow_controller().iter_tasks(None)):
            try:
                pm.push(task)
                all_passes[i] = task
            except Exception as exc:
                raise TypeError(f"Can't push {task}") from exc

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.rz(0.2, 0)
        circuit.cx(0, 1)

        callback = CallbackPrinter()
        out, _ = pm.run(circuit, callback)
        reference = full_pm.run(circuit)

        self.assertEqual(reference, out.to_circuit())
        self.assertTrue(
            {"HighLevelSynthesis", "UnitarySynthesis", "BasisTranslator"}.issubset(
                callback.pass_names
            )
        )
