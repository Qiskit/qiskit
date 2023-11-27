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
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Gate,
    Qubit,
    Clbit,
    Parameter,
    Operation,
    EquivalenceLibrary,
)
from qiskit.circuit.library import (
    SwapGate,
    CXGate,
    RZGate,
    PermutationGate,
    U3Gate,
    U2Gate,
    U1Gate,
    CU3Gate,
    CU1Gate,
)
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.converters import dag_to_circuit, circuit_to_dag, circuit_to_instruction
from qiskit.transpiler import PassManager, TranspilerError, CouplingMap, Target
from qiskit.transpiler.passes.basis import BasisTranslator
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.quantum_info import Operator
from qiskit.providers.fake_provider.fake_backend_v2 import FakeBackend5QV2
from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)


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

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        qc = QuantumCircuit(1)
        qc.id(0)
        return qc


class OpARepeatSynthesisPlugin(HighLevelSynthesisPlugin):
    """The repeat synthesis for opA"""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if "n" not in options.keys():
            return None

        qc = QuantumCircuit(1)
        for _ in range(options["n"]):
            qc.id(0)
        return qc


class OpBSimpleSynthesisPlugin(HighLevelSynthesisPlugin):
    """The simple synthesis for OpB"""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
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

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        num_swaps = options.get("num_swaps", self.num_swaps)

        qc = QuantumCircuit(2)
        for _ in range(num_swaps):
            qc.swap(0, 1)
        return qc


class OpAPluginNeedsCouplingMap(HighLevelSynthesisPlugin):
    """Synthesis plugins for OpA that needs a coupling map to be run."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if coupling_map is None:
            raise TranspilerError("Coupling map should be specified!")
        qc = QuantumCircuit(1)
        qc.id(0)
        return qc


class OpAPluginNeedsQubits(HighLevelSynthesisPlugin):
    """Synthesis plugins for OpA that needs ``qubits`` to be specified."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if qubits is None:
            raise TranspilerError("Qubits should be specified!")
        qc = QuantumCircuit(1)
        qc.id(0)
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
            "op_a.needs_coupling_map": OpAPluginNeedsCouplingMap,
            "op_a.needs_qubits": OpAPluginNeedsQubits,
        }

        self.plugins_by_op = {
            "op_a": ["default", "repeat", "needs_coupling_map", "needs_qubits"],
            "op_b": ["simple"],
        }

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

    def test_coupling_map_gets_passed_to_plugins(self):
        """Check that passing coupling map works correctly."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["needs_coupling_map"])
            pm_bad = PassManager([HighLevelSynthesis(hls_config=hls_config)])
            pm_good = PassManager(
                [
                    HighLevelSynthesis(
                        hls_config=hls_config, coupling_map=CouplingMap.from_line(qc.num_qubits)
                    )
                ]
            )

            # HighLevelSynthesis is initialized without a coupling map, but calling a plugin that
            # raises a TranspilerError without the coupling map.
            with self.assertRaises(TranspilerError):
                pm_bad.run(qc)

            # Now HighLevelSynthesis is initialized with a coupling map.
            pm_good.run(qc)

    def test_target_gets_passed_to_plugins(self):
        """Check that passing target (and constructing coupling map from the target)
        works correctly.
        """
        qc = QuantumCircuit(3)
        qc.append(OpA(), [0])

        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["needs_coupling_map"])
            pm_good = PassManager(
                [HighLevelSynthesis(hls_config=hls_config, target=FakeBackend5QV2().target)]
            )

            # HighLevelSynthesis is initialized with target.
            pm_good.run(qc)

    def test_qubits_get_passed_to_plugins(self):
        """Check that setting ``use_qubit_indices`` works correctly."""
        qc = self.create_circ()
        mock_plugin_manager = MockPluginManager
        with unittest.mock.patch(
            "qiskit.transpiler.passes.synthesis.high_level_synthesis.HighLevelSynthesisPluginManager",
            wraps=mock_plugin_manager,
        ):
            hls_config = HLSConfig(op_a=["needs_qubits"])
            pm_use_qubits_false = PassManager(
                [HighLevelSynthesis(hls_config=hls_config, use_qubit_indices=False)]
            )
            pm_use_qubits_true = PassManager(
                [HighLevelSynthesis(hls_config=hls_config, use_qubit_indices=True)]
            )

            # HighLevelSynthesis is initialized with use_qubit_indices=False, which means synthesis
            # plugin should see qubits=None and raise a transpiler error.
            with self.assertRaises(TranspilerError):
                pm_use_qubits_false.run(qc)

            # HighLevelSynthesis is initialized with use_qubit_indices=True, which means synthesis
            # plugin should see qubits and complete without errors.
            pm_use_qubits_true.run(qc)


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

    def test_definition_with_annotations(self):
        """Test annotated gates with definitions involving another annotated gate.
        Note that passing basis_gates makes the pass recursive.
        """
        qc = QuantumCircuit(4)
        lazy_gate1 = AnnotatedOperation(PermutationGate([3, 1, 0, 2]), InverseModifier())
        lazy_gate2 = AnnotatedOperation(SwapGate(), ControlModifier(2))
        qc.append(lazy_gate1, [0, 1, 2, 3])
        qc.append(lazy_gate2, [0, 1, 2, 3])
        custom_gate = qc.to_gate()
        lazy_gate3 = AnnotatedOperation(custom_gate, ControlModifier(2))
        circuit = QuantumCircuit(6)
        circuit.append(lazy_gate3, [0, 1, 2, 3, 4, 5])
        transpiled_circuit = HighLevelSynthesis(basis_gates=["cx", "u"])(circuit)
        self.assertEqual(Operator(circuit), Operator(transpiled_circuit))

    def test_definition_with_high_level_objects(self):
        """Test annotated gates with definitions involving annotations and
        high-level-objects."""
        def_circuit = QuantumCircuit(4)
        def_circuit.append(
            AnnotatedOperation(PermutationGate([1, 0]), ControlModifier(2)), [0, 1, 2, 3]
        )
        gate = def_circuit.to_gate()
        circuit = QuantumCircuit(6)
        circuit.append(gate, [0, 1, 2, 3])
        transpiled_circuit = HighLevelSynthesis()(circuit)
        expected_circuit = QuantumCircuit(6)
        expected_circuit.append(SwapGate().control(2), [0, 1, 2, 3])
        self.assertEqual(circuit, transpiled_circuit)

    def test_control_high_level_object(self):
        """Test synthesis of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, ControlModifier(1))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1, 2])
        backend = FakeBackend5QV2()
        qct = HighLevelSynthesis(target=backend.target)(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_transpile_control_high_level_object(self):
        """Test full transpilation of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, ControlModifier(1))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1, 2])
        backend = FakeBackend5QV2()
        qct = transpile(qc, target=backend.target)
        ops = qct.count_ops().keys()
        for op in ops:
            self.assertIn(op, ["u", "cx", "ecr", "measure"])

    def test_inverse_high_level_object(self):
        """Test synthesis of high level gates with inverse modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, InverseModifier())
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = FakeBackend5QV2()
        qct = HighLevelSynthesis(target=backend.target)(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_transpile_inverse_high_level_object(self):
        """Test synthesis of high level gates with inverse modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, InverseModifier())
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = FakeBackend5QV2()
        qct = transpile(qc, target=backend.target)
        ops = qct.count_ops().keys()
        for op in ops:
            self.assertIn(op, ["u", "cx", "ecr", "measure"])

    def test_power_high_level_object(self):
        """Test synthesis of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, PowerModifier(3))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = FakeBackend5QV2()
        qct = HighLevelSynthesis(target=backend.target)(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_transpile_power_high_level_object(self):
        """Test full transpilation of high level gates with control modifier."""
        linear_circuit = QuantumCircuit(2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 0)
        linear_function = LinearFunction(linear_circuit)
        annotated_linear_function = AnnotatedOperation(linear_function, PowerModifier(3))
        qc = QuantumCircuit(3)
        qc.append(annotated_linear_function, [0, 1])
        backend = FakeBackend5QV2()
        qct = transpile(qc, target=backend.target)
        ops = qct.count_ops().keys()
        for op in ops:
            self.assertIn(op, ["u", "cx", "ecr", "measure"])


class TestUnrollerCompatability(QiskitTestCase):
    """Tests backward compatibility with the UnrollCustomDefinitions pass.

    Duplicate of TestUnrollerCompatability from test.python.transpiler.test_basis_translator,
    with UnrollCustomDefinitions replaced by HighLevelSynthesis.
    """

    def test_basic_unroll(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u2"])
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["u2"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u2")

    def test_unroll_toffoli(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(
            equivalence_library=std_eqlib, basis_gates=["h", "t", "tdg", "cx"]
        )
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["h", "t", "tdg", "cx"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_basic_unroll_min_qubits(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u2"], min_qubits=3)
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["u2"], min_qubits=3)
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "h")

    def test_unroll_toffoli_min_qubits(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        circuit.sx(qr1[0])
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(
            equivalence_library=std_eqlib, basis_gates=["h", "t", "tdg", "cx"], min_qubits=3
        )
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["h", "t", "tdg", "cx"], min_qubits=3)
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 16)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx", "sx"])

    def test_unroll_1q_chain_conditional(self):
        """Test unroll chain of 1-qubit gates interrupted by conditional."""

        #     ┌───┐┌─────┐┌───┐┌───┐┌─────────┐┌─────────┐┌─────────┐┌─┐ ┌───┐  ┌───┐ »
        # qr: ┤ H ├┤ Tdg ├┤ Z ├┤ T ├┤ Ry(0.5) ├┤ Rz(0.3) ├┤ Rx(0.1) ├┤M├─┤ X ├──┤ Y ├─»
        #     └───┘└─────┘└───┘└───┘└─────────┘└─────────┘└─────────┘└╥┘ └─╥─┘  └─╥─┘ »
        #                                                             ║ ┌──╨──┐┌──╨──┐»
        # cr: 1/══════════════════════════════════════════════════════╩═╡ 0x1 ╞╡ 0x1 ╞»
        #                                                             0 └─────┘└─────┘»
        # «       ┌───┐
        # «  qr: ─┤ Z ├─
        # «       └─╥─┘
        # «      ┌──╨──┐
        # «cr: 1/╡ 0x1 ╞
        # «      └─────┘
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.tdg(qr)
        circuit.z(qr)
        circuit.t(qr)
        circuit.ry(0.5, qr)
        circuit.rz(0.3, qr)
        circuit.rx(0.1, qr)
        circuit.measure(qr, cr)
        circuit.x(qr).c_if(cr, 1)
        circuit.y(qr).c_if(cr, 1)
        circuit.z(qr).c_if(cr, 1)
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u1", "u2", "u3"])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ["u1", "u2", "u3"])
        unrolled_dag = pass_.run(dag)

        # Pick up -1 * 0.3 / 2 global phase for one RZ -> U1.
        #
        # global phase: 6.1332
        #     ┌─────────┐┌──────────┐┌───────┐┌─────────┐┌─────────────┐┌─────────┐»
        # qr: ┤ U2(0,π) ├┤ U1(-π/4) ├┤ U1(π) ├┤ U1(π/4) ├┤ U3(0.5,0,0) ├┤ U1(0.3) ├»
        #     └─────────┘└──────────┘└───────┘└─────────┘└─────────────┘└─────────┘»
        # cr: 1/═══════════════════════════════════════════════════════════════════»
        #                                                                          »
        # «      ┌──────────────────┐┌─┐┌───────────┐┌───────────────┐┌───────┐
        # «  qr: ┤ U3(0.1,-π/2,π/2) ├┤M├┤ U3(π,0,π) ├┤ U3(π,π/2,π/2) ├┤ U1(π) ├
        # «      └──────────────────┘└╥┘└─────╥─────┘└───────╥───────┘└───╥───┘
        # «                           ║    ┌──╨──┐        ┌──╨──┐      ┌──╨──┐
        # «cr: 1/═════════════════════╩════╡ 0x1 ╞════════╡ 0x1 ╞══════╡ 0x1 ╞═
        # «                           0    └─────┘        └─────┘      └─────┘
        ref_circuit = QuantumCircuit(qr, cr, global_phase=-0.3 / 2)
        ref_circuit.append(U2Gate(0, np.pi), [qr[0]])
        ref_circuit.append(U1Gate(-np.pi / 4), [qr[0]])
        ref_circuit.append(U1Gate(np.pi), [qr[0]])
        ref_circuit.append(U1Gate(np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0.5, 0, 0), [qr[0]])
        ref_circuit.append(U1Gate(0.3), [qr[0]])
        ref_circuit.append(U3Gate(0.1, -np.pi / 2, np.pi / 2), [qr[0]])
        ref_circuit.measure(qr[0], cr[0])
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [qr[0]]).c_if(cr, 1)
        ref_circuit.append(U3Gate(np.pi, np.pi / 2, np.pi / 2), [qr[0]]).c_if(cr, 1)
        ref_circuit.append(U1Gate(np.pi), [qr[0]]).c_if(cr, 1)
        ref_dag = circuit_to_dag(ref_circuit)

        self.assertEqual(unrolled_dag, ref_dag)

    def test_unroll_no_basis(self):
        """Test when a given gate has no decompositions."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=[])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, [])

        with self.assertRaises(QiskitError):
            pass_.run(dag)

    def test_unroll_all_instructions(self):
        """Test unrolling a circuit containing all standard instructions."""

        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.crx(0.5, qr[1], qr[2])
        circuit.cry(0.5, qr[1], qr[2])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.ch(qr[0], qr[2])
        circuit.crz(0.5, qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.append(CU1Gate(0.1), [qr[0], qr[2]])
        circuit.append(CU3Gate(0.2, 0.1, 0.0), [qr[1], qr[2]])
        circuit.cx(qr[1], qr[0])
        circuit.cy(qr[1], qr[2])
        circuit.cz(qr[2], qr[0])
        circuit.h(qr[1])
        circuit.rx(0.1, qr[0])
        circuit.ry(0.2, qr[1])
        circuit.rz(0.3, qr[2])
        circuit.rzz(0.6, qr[1], qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.t(qr[2])
        circuit.tdg(qr[0])
        circuit.append(U1Gate(0.1), [qr[1]])
        circuit.append(U2Gate(0.2, -0.1), [qr[0]])
        circuit.append(U3Gate(0.3, 0.0, -0.1), [qr[2]])
        circuit.x(qr[2])
        circuit.y(qr[1])
        circuit.z(qr[0])
        # circuit.snapshot('0')
        # circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u3", "cx", "id"])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ["u3", "cx", "id"])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0.25, -np.pi / 2, 0), [qr[2]])
        ref_circuit.append(U3Gate(0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[1]])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 2), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, 0.25), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, -0.25), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[1]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[2]])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[0]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.1, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[0]])
        ref_circuit.append(U3Gate(0.1, 0.1, 0), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 2), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[1]])
        ref_circuit.append(U3Gate(0.2, 0, 0), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[2]])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(np.pi / 2, 0, np.pi), [qr[0]])
        ref_circuit.append(U3Gate(0.1, -np.pi / 2, np.pi / 2), [qr[0]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.6), [qr[0]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, np.pi / 2), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(np.pi / 2, 0.2, -0.1), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, np.pi), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -np.pi / 2), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, 0.3), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[1])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, 0.1), [qr[1]])
        ref_circuit.append(U3Gate(np.pi, np.pi / 2, np.pi / 2), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, np.pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(0.3, 0.0, -0.1), [qr[2]])
        ref_circuit.append(U3Gate(np.pi, 0, np.pi), [qr[2]])
        # ref_circuit.snapshot('0')
        # ref_circuit.measure(qr, cr)
        # ref_dag = circuit_to_dag(ref_circuit)

        self.assertTrue(Operator(dag_to_circuit(unrolled_dag)).equiv(ref_circuit))

    def test_simple_unroll_parameterized_without_expressions(self):
        """Verify unrolling parameterized gates without expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.rz(theta, qr[0])
        dag = circuit_to_dag(qc)

        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["u1", "cx"])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ["u1", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-theta / 2)
        expected.append(U1Gate(theta), [qr[0]])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_simple_unroll_parameterized_with_expressions(self):
        """Verify unrolling parameterized gates with expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        qc.rz(sum_, qr[0])
        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ["p", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-sum_ / 2)
        expected.p(sum_, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_definition_unroll_parameterized(self):
        """Verify that unrolling complex gates with parameters does not raise."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.cp(theta, qr[1], qr[0])
        qc.cp(theta * theta, qr[0], qr[1])
        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=std_eqlib, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(std_eqlib, ["p", "cx"]).run(dag)

        self.assertEqual(out_dag.count_ops(), {"p": 6, "cx": 4})

    def test_unrolling_parameterized_composite_gates(self):
        """Verify unrolling circuits with parameterized composite gates."""
        mock_sel = EquivalenceLibrary(base=std_eqlib)

        qr1 = QuantumRegister(2)
        subqc = QuantumCircuit(qr1)

        theta = Parameter("theta")

        subqc.rz(theta, qr1[0])
        subqc.cx(qr1[0], qr1[1])
        subqc.rz(theta, qr1[1])

        # Expanding across register with shared parameter
        qr2 = QuantumRegister(4)
        qc = QuantumCircuit(qr2)

        sub_instr = circuit_to_instruction(subqc, equivalence_library=mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=mock_sel, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ["p", "cx"]).run(dag)

        # Pick up -1 * theta / 2 global phase four twice (once for each RZ -> P
        # in each of the two sub_instr instructions).
        expected = QuantumCircuit(qr2, global_phase=-1 * 4 * theta / 2.0)
        expected.p(theta, qr2[0])
        expected.p(theta, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.p(theta, qr2[1])
        expected.p(theta, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)

        # Expanding across register with shared parameter
        qc = QuantumCircuit(qr2)

        phi = Parameter("phi")
        gamma = Parameter("gamma")

        sub_instr = circuit_to_instruction(subqc, {theta: phi}, mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        sub_instr = circuit_to_instruction(subqc, {theta: gamma}, mock_sel)
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = HighLevelSynthesis(equivalence_library=mock_sel, basis_gates=["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ["p", "cx"]).run(dag)

        expected = QuantumCircuit(qr2, global_phase=-1 * (2 * phi + 2 * gamma) / 2.0)
        expected.p(phi, qr2[0])
        expected.p(gamma, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.p(phi, qr2[1])
        expected.p(gamma, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)


class TestGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self):
        super().__init__("tg", 1, [])


class TestCompositeGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self):
        super().__init__("tcg", 1, [])


class TestUnrollCustomDefinitionsCompatibility(QiskitTestCase):
    """Tests backward compatibility with the UnrollCustomDefinitions pass.

    Duplicate of TestUnrollCustomDefinitions from test.python.transpiler.test_unroll_custom_definitions,
    with UnrollCustomDefinitions replaced by HighLevelSynthesis.
    """

    def test_dont_unroll_a_gate_in_eq_lib(self):
        """Verify we don't unroll a gate found in equivalence_library."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx"]).run(dag)

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_dont_unroll_a_gate_in_basis_gates(self):
        """Verify we don't unroll a gate in basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx", "tg"]).run(
            dag
        )

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_raise_for_opaque_not_in_eq_lib(self):
        """Verify we raise for an opaque gate not in basis_gates or eq_lib."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        with self.assertRaisesRegex(QiskitError, "unable to synthesize"):
            HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx"]).run(dag)

    def test_unroll_gate_until_reach_basis_gates(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestCompositeGate()
        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(q)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx", "tg"]).run(
            dag
        )

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_twice_until_we_get_to_eqlib(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(base_gate, equiv)

        gate = TestCompositeGate()

        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["u3", "cx"]).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_if_else(self):
        """Test that a simple if-else unrolls correctly."""
        eq_lib = EquivalenceLibrary()

        equiv = QuantumCircuit(1)
        equiv.h(0)
        eq_lib.add_equivalence(TestGate(), equiv)

        equiv = QuantumCircuit(1)
        equiv.z(0)
        eq_lib.add_equivalence(TestCompositeGate(), equiv)

        pass_ = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["h", "z", "cx"])

        true_body = QuantumCircuit(1)
        true_body.h(0)
        true_body.append(TestGate(), [0])
        false_body = QuantumCircuit(1)
        false_body.append(TestCompositeGate(), [0])

        test = QuantumCircuit(1, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), true_body, false_body, [0], [])

        expected = QuantumCircuit(1, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), pass_(true_body), pass_(false_body), [0], [])

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that the unroller recurses into nested control flow."""
        eq_lib = EquivalenceLibrary()
        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)
        eq_lib.add_equivalence(base_gate, equiv)
        base_gate = TestCompositeGate()
        equiv = QuantumCircuit(1)
        equiv.z(0)
        eq_lib.add_equivalence(base_gate, equiv)

        pass_ = HighLevelSynthesis(equivalence_library=eq_lib, basis_gates=["h", "z", "cx"])

        qubit = Qubit()
        clbit = Clbit()

        for_body = QuantumCircuit(1)
        for_body.append(TestGate(), [0], [])

        while_body = QuantumCircuit(1)
        while_body.append(TestCompositeGate(), [0], [])

        true_body = QuantumCircuit([qubit, clbit])
        true_body.while_loop((clbit, True), while_body, [0], [])

        test = QuantumCircuit([qubit, clbit])
        test.for_loop(range(2), None, for_body, [0], [])
        test.if_else((clbit, True), true_body, None, [0], [0])

        expected_if_body = QuantumCircuit([qubit, clbit])
        expected_if_body.while_loop((clbit, True), pass_(while_body), [0], [])
        expected = QuantumCircuit([qubit, clbit])
        expected.for_loop(range(2), None, pass_(for_body), [0], [])
        expected.if_else(range(2), pass_(expected_if_body), None, [0], [0])

        self.assertEqual(pass_(test), expected)

    def test_dont_unroll_a_gate_in_basis_gates_with_target(self):
        """Verify we don't unroll a gate in basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        dag = circuit_to_dag(qc)
        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())
        target.add_instruction(TestGate())

        out = HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

        expected = qc.copy()
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_raise_for_opaque_not_in_eq_lib_target_with_target(self):
        """Verify we raise for an opaque gate not in basis_gates or eq_lib."""
        eq_lib = EquivalenceLibrary()

        gate = TestGate()
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())

        dag = circuit_to_dag(qc)
        with self.assertRaisesRegex(QiskitError, "unable to synthesize"):
            HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

    def test_unroll_gate_until_reach_basis_gates_with_target(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        gate = TestCompositeGate()
        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(q)
        qc.append(gate, [0])

        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())
        target.add_instruction(TestGate())

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_twice_until_we_get_to_eqlib_with_target(self):
        """Verify we unroll gates until we hit basis_gates."""
        eq_lib = EquivalenceLibrary()

        base_gate = TestGate()
        equiv = QuantumCircuit(1)
        equiv.h(0)

        eq_lib.add_equivalence(base_gate, equiv)

        gate = TestCompositeGate()

        q = QuantumRegister(1, "q")
        gate.definition = QuantumCircuit(q)
        gate.definition.append(TestGate(), [q[0]], [])

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        target = Target(num_qubits=1)
        target.add_instruction(U3Gate(Parameter("a"), Parameter("b"), Parameter("c")))
        target.add_instruction(CXGate())

        dag = circuit_to_dag(qc)
        out = HighLevelSynthesis(equivalence_library=eq_lib, target=target).run(dag)

        expected = QuantumCircuit(1)
        expected.append(TestGate(), [0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out, expected_dag)

    def test_unroll_empty_definition(self):
        """Test that a gate with no operations can be unrolled."""
        qc = QuantumCircuit(2)
        qc.append(QuantumCircuit(2).to_gate(), [0, 1], [])
        pass_ = HighLevelSynthesis(equivalence_library=EquivalenceLibrary(), basis_gates=["u"])
        expected = QuantumCircuit(2)
        self.assertEqual(pass_(qc), expected)

    def test_unroll_empty_definition_with_phase(self):
        """Test that a gate with no operations but with a global phase can be unrolled."""
        qc = QuantumCircuit(2)
        qc.append(QuantumCircuit(2, global_phase=0.5).to_gate(), [0, 1], [])
        pass_ = HighLevelSynthesis(equivalence_library=EquivalenceLibrary(), basis_gates=["u"])
        expected = QuantumCircuit(2, global_phase=0.5)
        self.assertEqual(pass_(qc), expected)


if __name__ == "__main__":
    unittest.main()
