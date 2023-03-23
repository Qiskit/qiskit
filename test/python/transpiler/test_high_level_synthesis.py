# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests the interface for HighLevelSynthesis transpiler pass.
"""


import unittest.mock
import numpy as np
from qiskit.circuit import QuantumCircuit, Operation
from qiskit.circuit.library import SwapGate, CXGate, RZGate, PermutationGate
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.quantum_info import Operator


# In what follows, we create two simple operations OpA and OpB, that potentially mimic
# higher-level objects written by a user.
# For OpA we define two synthesis methods:
#  - "default", which does not require any additional parameters, and
#  - "repeat", which requires a parameter "n" s.t. the "repeat" returns None when "n" is not
#              specified.
# For OpB we define a single synthesis method:
#  - "simple", which does not require any additional parameters.
# Note that OpB does not have a "default" method specified.
# Finally, we will mock the HighLevelSynthesisPluginManager by a dummy class that implements
# a similar functionality, but without depending on the stevedore extension manager.


class OpA(Operation):
    """A simple operation."""

    @property
    def name(self):
        return "op_a"

    @property
    def num_qubits(self):
        return 1

    @property
    def num_clbits(self):
        return 0


class OpB(Operation):
    """Another simple operation."""

    @property
    def name(self):
        return "op_b"

    @property
    def num_qubits(self):
        return 2

    @property
    def num_clbits(self):
        return 0


class OpADefaultSynthesisPlugin(HighLevelSynthesisPlugin):
    """The default synthesis for opA"""

    def run(self, high_level_object, **options):
        qc = QuantumCircuit(1)
        qc.id(0)
        return qc


class OpARepeatSynthesisPlugin(HighLevelSynthesisPlugin):
    """The repeat synthesis for opA"""

    def run(self, high_level_object, **options):
        if "n" not in options.keys():
            return None

        qc = QuantumCircuit(1)
        for _ in range(options["n"]):
            qc.id(0)
        return qc


class OpBSimpleSynthesisPlugin(HighLevelSynthesisPlugin):
    """The simple synthesis for OpB"""

    def run(self, high_level_object, **options):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        return qc


class OpBAnotherSynthesisPlugin(HighLevelSynthesisPlugin):
    """Another synthesis plugin for OpB objects.

    This plugin is not registered in MockPluginManager, and is used to
    illustrate the alternative construction mechanism using raw classes.
    """

    def __init__(self, num_swaps=1):
        self.num_swaps = num_swaps

    def run(self, high_level_object, **options):
        num_swaps = options.get("num_swaps", self.num_swaps)

        qc = QuantumCircuit(2)
        for _ in range(num_swaps):
            qc.swap(0, 1)
        return qc


class MockPluginManager:
    """Mocks the functionality of HighLevelSynthesisPluginManager,
    without actually depending on the stevedore extension manager.
    """

    def __init__(self):
        self.plugins = {
            "op_a.default": OpADefaultSynthesisPlugin,
            "op_a.repeat": OpARepeatSynthesisPlugin,
            "op_b.simple": OpBSimpleSynthesisPlugin,
        }

        self.plugins_by_op = {"op_a": ["default", "repeat"], "op_b": ["simple"]}

    def method_names(self, op_name):
        """Returns plugin methods for op_name."""
        if op_name in self.plugins_by_op.keys():
            return self.plugins_by_op[op_name]
        else:
            return []

    def method(self, op_name, method_name):
        """Returns the plugin for ``op_name`` and ``method_name``."""
        plugin_name = op_name + "." + method_name
        return self.plugins[plugin_name]()


class TestHighLevelSynthesisInterface(QiskitTestCase):
    """Tests for the synthesis plugin interface."""

    def create_circ(self):
        """Create a simple circuit used for tests with two OpA gates and one OpB gate."""
        qc = QuantumCircuit(3)
        qc.append(OpA(), [0])
        qc.append(OpB(), [0, 1])
        qc.append(OpA(), [2])
        return qc

    def test_no_config(self):
        """Check the default behavior of HighLevelSynthesis, without
        HighLevelSynthesisConfig specified. In this case, the default
        synthesis methods should be used when defined. OpA has such a
        method, and OpB does not."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            pm = PassManager([HighLevelSynthesis()])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # OpA's default method replaces by "id", OpB has no default method
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())
            self.assertEqual(ops["op_b"], 1)

    def test_default_config(self):
        """Check the default behavior of HighLevelSynthesis, with
        the default HighLevelSynthesisConfig specified. The behavior should
        be the same as without config."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig()
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # OpA's default method replaces by "id", OpB has no default method
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())
            self.assertEqual(ops["op_b"], 1)

    def test_non_default_config(self):
        """Check the default behavior of HighLevelSynthesis, specifying
        non-default synthesis methods for OpA and for OpB.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[("repeat", {"n": 2})], op_b=[("simple", {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            self.assertNotIn("op_a", ops.keys())
            self.assertNotIn("op_b", ops.keys())
            self.assertEqual(ops["id"], 4)
            self.assertEqual(ops["cx"], 1)

    def test_synthesis_returns_none(self):
        """Check that when synthesis method is specified but returns None,
        the operation does not get synthesized.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[("repeat", {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertIn("op_a", ops.keys())
            self.assertIn("op_b", ops.keys())

    def test_use_default_on_unspecified_is_false(self):
        """Check that when use_default_on_unspecified is False, the default synthesis
        method is not applied.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(use_default_on_unspecified=False)
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertIn("op_a", ops.keys())
            self.assertIn("op_b", ops.keys())

    def test_use_default_on_unspecified_is_true(self):
        """Check that when use_default_on_unspecified is True (which should be the default
        value), the default synthesis method gets applied.
        OpA has such a method, and OpB does not."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            pm = PassManager([HighLevelSynthesis()])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # OpA's default method replaces by "id", OpB has no default method
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())
            self.assertEqual(ops["op_b"], 1)

    def test_skip_synthesis_with_empty_methods_list(self):
        """Check that when synthesis config is specified, but an operation
        is given an empty list of methods, it is not synthesized.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertIn("op_a", ops.keys())
            self.assertIn("op_b", ops.keys())

    def test_multiple_methods(self):
        """Check that when there are two synthesis methods specified,
        and the first returns None, then the second method gets used.
        """
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=[("repeat", {}), ("default", {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())

    def test_multiple_methods_short_form(self):
        """Check that when there are two synthesis methods specified,
        and the first returns None, then the second method gets used.
        In this example, the list of methods is specified without
        explicitly listing empty argument lists.
        """

        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["repeat", "default"])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            # The repeat method for OpA without "n" specified returns None.
            self.assertNotIn("op_a", ops.keys())
            self.assertEqual(ops["id"], 2)
            self.assertIn("op_b", ops.keys())

    def test_synthesis_using_alternate_form(self):
        """Test alternative form of specifying synthesis methods."""

        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            # synthesis using raw class, without extension manager
            plugin = OpBAnotherSynthesisPlugin(num_swaps=6)
            hls_config = HLSConfig(op_b=[(plugin, {})])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            self.assertEqual(ops["swap"], 6)

    def test_synthesis_using_alternate_short_form(self):
        """Test alternative form of specifying synthesis methods."""

        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            # synthesis using raw class, without extension manager
            plugin = OpBAnotherSynthesisPlugin(num_swaps=6)
            hls_config = HLSConfig(op_b=[plugin])
            pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            tqc = pm.run(qc)
            ops = tqc.count_ops()
            self.assertEqual(ops["swap"], 6)


class TestHighLevelSynthesisModifiers(QiskitTestCase):
    """Tests for high-level-synthesis pass."""

    def test_control_basic_gates(self):
        """Test lazy control synthesis of basic gates (each has its class ``control`` method)."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        lazy_gate2 = AnnotatedOperation(CXGate(), ControlModifier(1))
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), ControlModifier(1))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1, 2, 3])
        circuit.append(lazy_gate2, [0, 1, 2])
        circuit.append(lazy_gate3, [2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        controlled_gate1 = SwapGate().control(2)
        controlled_gate2 = CXGate().control(1)
        controlled_gate3 = RZGate(np.pi / 4).control(1)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(controlled_gate1, [0, 1, 2, 3])
        expected_circuit.append(controlled_gate2, [0, 1, 2])
        expected_circuit.append(controlled_gate3, [2, 3])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_control_custom_gates(self):
        """Test lazy control synthesis of custom gates (which inherits ``control`` method from
        ``Gate``).
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        gate = qc.to_gate()
        circuit = QuantumCircuit(4)
        circuit.append(AnnotatedOperation(gate, ControlModifier(2)), [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(gate.control(2), [0, 1, 2, 3])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_control_clifford(self):
        """Test lazy control synthesis of Clifford objects (no ``control`` method defined)."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(4)
        circuit.append(AnnotatedOperation(cliff, ControlModifier(2)), [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(cliff.to_instruction().control(2), [0, 1, 2, 3])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_multiple_controls(self):
        """Test lazy controlled synthesis with multiple control modifiers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), [ControlModifier(2), ControlModifier(1)])
        circuit = QuantumCircuit(5)
        circuit.append(lazy_gate1, [0, 1, 2, 3, 4])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(5)
        expected_circuit.append(SwapGate().control(2).control(1), [0, 1, 2, 3, 4])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_controls(self):
        """Test lazy controlled synthesis of nested lazy gates."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, ControlModifier(1))
        circuit = QuantumCircuit(5)
        circuit.append(lazy_gate2, [0, 1, 2, 3, 4])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(5)
        expected_circuit.append(SwapGate().control(2).control(1), [0, 1, 2, 3, 4])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_controls_permutation(self):
        """Test lazy controlled synthesis of ``PermutationGate`` with nested lazy gates.
        Note that ``PermutationGate`` currently does not have definition."""
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), ControlModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, ControlModifier(1))
        circuit = QuantumCircuit(7)
        circuit.append(lazy_gate2, [0, 1, 2, 3, 4, 5, 6])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_inverse_basic_gates(self):
        """Test lazy inverse synthesis of basic gates (each has its class ``control`` method)."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), InverseModifier())
        lazy_gate2 = AnnotatedOperation(CXGate(), InverseModifier())
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), InverseModifier())
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 2])
        circuit.append(lazy_gate2, [0, 1])
        circuit.append(lazy_gate3, [2])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        inverse_gate1 = SwapGate().inverse()
        inverse_gate2 = CXGate().inverse()
        inverse_gate3 = RZGate(np.pi / 4).inverse()
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(inverse_gate1, [0, 2])
        expected_circuit.append(inverse_gate2, [0, 1])
        expected_circuit.append(inverse_gate3, [2])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_inverse_custom_gates(self):
        """Test lazy control synthesis of custom gates (which inherits ``inverse`` method from
        ``Gate``).
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        gate = qc.to_gate()
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(gate, InverseModifier()), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(gate.inverse(), [0, 1])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_inverse_clifford(self):
        """Test lazy inverse synthesis of Clifford objects (no ``inverse`` method defined)."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(cliff, InverseModifier()), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(cliff.to_instruction().inverse(), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_two_inverses(self):
        """Test lazy controlled synthesis with multiple inverse modifiers (even)."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), [InverseModifier(), InverseModifier()])
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate1, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate().inverse().inverse(), [0, 1])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_three_inverses(self):
        """Test lazy controlled synthesis with multiple inverse modifiers (odd)."""
        lazy_gate1 = AnnotatedOperation(
            RZGate(np.pi / 4), [InverseModifier(), InverseModifier(), InverseModifier()]
        )
        circuit = QuantumCircuit(1)
        circuit.append(lazy_gate1, [0])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(1)
        expected_circuit.append(RZGate(np.pi / 4).inverse().inverse().inverse(), [0])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_inverses(self):
        """Test lazy synthesis with nested lazy gates."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), InverseModifier())
        lazy_gate2 = AnnotatedOperation(lazy_gate1, InverseModifier())
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate2, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate(), [0, 1])
        self.assertEqual(transpiled_circuit, expected_circuit)

    def test_nested_inverses_permutation(self):
        """Test lazy controlled synthesis of ``PermutationGate`` with nested lazy gates.
        Note that ``PermutationGate`` currently does not have definition."""
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), InverseModifier())
        lazy_gate2 = AnnotatedOperation(lazy_gate1, InverseModifier())
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate2, [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_power_posint_basic_gates(self):
        """Test lazy power synthesis of basic gates with positive and zero integer powers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), PowerModifier(2))
        lazy_gate2 = AnnotatedOperation(CXGate(), PowerModifier(1))
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), PowerModifier(3))
        lazy_gate4 = AnnotatedOperation(CXGate(), PowerModifier(0))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1])
        circuit.append(lazy_gate2, [1, 2])
        circuit.append(lazy_gate3, [3])
        circuit.append(lazy_gate4, [2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(SwapGate(), [0, 1])
        expected_circuit.append(SwapGate(), [0, 1])
        expected_circuit.append(CXGate(), [1, 2])
        expected_circuit.append(RZGate(np.pi / 4), [3])
        expected_circuit.append(RZGate(np.pi / 4), [3])
        expected_circuit.append(RZGate(np.pi / 4), [3])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_negint_basic_gates(self):
        """Test lazy power synthesis of basic gates with negative integer powers."""
        lazy_gate1 = AnnotatedOperation(CXGate(), PowerModifier(-1))
        lazy_gate2 = AnnotatedOperation(RZGate(np.pi / 4), PowerModifier(-3))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1])
        circuit.append(lazy_gate2, [2])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(CXGate(), [0, 1])
        expected_circuit.append(RZGate(-np.pi / 4), [2])
        expected_circuit.append(RZGate(-np.pi / 4), [2])
        expected_circuit.append(RZGate(-np.pi / 4), [2])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_float_basic_gates(self):
        """Test lazy power synthesis of basic gates with floating-point powers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), PowerModifier(0.5))
        lazy_gate2 = AnnotatedOperation(CXGate(), PowerModifier(0.2))
        lazy_gate3 = AnnotatedOperation(RZGate(np.pi / 4), PowerModifier(-0.25))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate1, [0, 1])
        circuit.append(lazy_gate2, [1, 2])
        circuit.append(lazy_gate3, [3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(SwapGate().power(0.5), [0, 1])
        expected_circuit.append(CXGate().power(0.2), [1, 2])
        expected_circuit.append(RZGate(np.pi / 4).power(-0.25), [3])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_custom_gates(self):
        """Test lazy power synthesis of custom gates with positive integer powers."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        gate = qc.to_gate()
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(gate, PowerModifier(3)), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(gate, [0, 1])
        expected_circuit.append(gate, [0, 1])
        expected_circuit.append(gate, [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_posint_clifford(self):
        """Test lazy power synthesis of Clifford objects with positive integer power."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(cliff, PowerModifier(3)), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(cliff.to_instruction().power(3), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_power_float_clifford(self):
        """Test lazy power synthesis of Clifford objects with floating-point power."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)
        cliff = Clifford(qc)
        circuit = QuantumCircuit(2)
        circuit.append(AnnotatedOperation(cliff, PowerModifier(-0.5)), [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(cliff.to_instruction().power(-0.5), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_multiple_powers(self):
        """Test lazy controlled synthesis with multiple power modifiers."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), [PowerModifier(2), PowerModifier(-1)])
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate1, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate().power(-2), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_nested_powers(self):
        """Test lazy synthesis with nested lazy gates."""
        lazy_gate1 = AnnotatedOperation(SwapGate(), PowerModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, PowerModifier(-1))
        circuit = QuantumCircuit(2)
        circuit.append(lazy_gate2, [0, 1])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(2)
        expected_circuit.append(SwapGate().power(-2), [0, 1])
        self.assertEqual(Operator(transpiled_circuit), Operator(expected_circuit))

    def test_nested_powers_permutation(self):
        """Test lazy controlled synthesis of ``PermutationGate`` with nested lazy gates.
        Note that ``PermutationGate`` currently does not have definition."""
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), PowerModifier(2))
        lazy_gate2 = AnnotatedOperation(lazy_gate1, PowerModifier(-1))
        circuit = QuantumCircuit(4)
        circuit.append(lazy_gate2, [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_multiple_modifiers(self):
        """Test involving gates with different modifiers."""
        qc = QuantumCircuit(4)
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), InverseModifier())
        lazy_gate2 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        qc.append(lazy_gate1, [0, 1, 2, 3])
        qc.append(lazy_gate2, [0, 1, 2, 3])
        custom_gate = qc.to_gate()
        lazy_gate3 = AnnotatedOperation(custom_gate, ControlModifier(2))
        circuit = QuantumCircuit(6)
        circuit.append(lazy_gate3, [0, 1, 2, 3, 4, 5])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_reordered_modifiers(self):
        """Test involving gates with different modifiers."""
        lazy_gate1 = AnnotatedOperation(
            PermutationGate([3, 1, 0, 2]), [InverseModifier(), ControlModifier(2), PowerModifier(3)]
        )
        lazy_gate2 = AnnotatedOperation(
            PermutationGate([3, 1, 0, 2]), [PowerModifier(3), ControlModifier(2), InverseModifier()]
        )
        qc1 = QuantumCircuit(6)
        qc1.append(lazy_gate1, [0, 1, 2, 3, 4, 5])
        qc2 = QuantumCircuit(6)
        qc2.append(lazy_gate2, [0, 1, 2, 3, 4, 5])
        self.assertEqual(Operator(qc1), Operator(qc2))
        transpiled1 = HighLevelSynthesis()(qc1)
        transpiled2 = HighLevelSynthesis()(qc2)
        self.assertEqual(Operator(transpiled1), Operator(transpiled2))
        self.assertEqual(Operator(qc1), Operator(transpiled1))


if __name__ == "__main__":
    unittest.main()
