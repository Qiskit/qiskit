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
import itertools
import unittest.mock

from qiskit.circuit import QuantumCircuit, Operation
from qiskit.circuit.library import PermutationGate
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager, TranspilerError, CouplingMap
from qiskit.transpiler.passes.synthesis.plugin import (
    HighLevelSynthesisPlugin,
    HighLevelSynthesisPluginManager,
)
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.providers.fake_provider.fake_backend_v2 import FakeBackend5QV2
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


class TestHighLeverSynthesisInterface(QiskitTestCase):
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
        qc = self.create_circ()
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


class TestTokenSwapperPermutationPlugin(QiskitTestCase):
    """Tests for the token swapper plugin for synthesizing permutation gates."""

    def test_token_swapper_in_known_plugin_names(self):
        """Test that "token_swapper" is an available synthesis plugin for permutation gates."""
        self.assertIn(
            "token_swapper", HighLevelSynthesisPluginManager().method_names("permutation")
        )

    def test_abstract_synthesis(self):
        """Test abstract synthesis of a permutation gate (either the coupling map or the set
        of qubits over which the permutation is defined is not specified).
        """

        # Permutation gate
        # 4->0, 6->1, 3->2, 7->3, 1->4, 2->5, 0->6, 5->7
        perm = PermutationGate([4, 6, 3, 7, 1, 2, 0, 5])

        # Circuit with permutation gate
        qc = QuantumCircuit(8)
        qc.append(perm, range(8))

        # Synthesize circuit using the token swapper plugin
        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10, "seed": 1})])
        qc_transpiled = PassManager(HighLevelSynthesis(synthesis_config)).run(qc)

        # Construct the expected quantum circuit
        # From the description below we can see that
        #   0->6, 1->4, 2->5, 3->2, 4->0, 5->2->3->7, 6->0->4->1, 7->3
        qc_expected = QuantumCircuit(8)
        qc_expected.swap(2, 5)
        qc_expected.swap(0, 6)
        qc_expected.swap(2, 3)
        qc_expected.swap(0, 4)
        qc_expected.swap(1, 4)
        qc_expected.swap(3, 7)

        self.assertEqual(qc_transpiled, qc_expected)

    def test_concrete_synthesis(self):
        """Test concrete synthesis of a permutation gate (we have both the coupling map and the
        set of qubits over which the permutation gate is defined; moreover, the coupling map may
        have more qubits than the permutation gate).
        """

        # Permutation gate
        perm = PermutationGate([0, 1, 4, 3, 2])

        # Circuit with permutation gate
        qc = QuantumCircuit(8)
        qc.append(perm, [3, 4, 5, 6, 7])

        coupling_map = CouplingMap.from_ring(8)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        qc_transpiled = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        ).run(qc)

        qc_expected = QuantumCircuit(8)
        qc_expected.swap(6, 7)
        qc_expected.swap(5, 6)
        qc_expected.swap(6, 7)
        self.assertEqual(qc_transpiled, qc_expected)

    def test_concrete_synthesis_over_disconnected_qubits(self):
        """Test concrete synthesis of a permutation gate over a disconnected set of qubits.
        In this case the plugin should return `None` and `HighLevelSynthesis`
        should not change the original circuit.
        """

        # Permutation gate
        perm = PermutationGate([4, 3, 2, 1, 0])

        # Circuit with permutation gate
        qc = QuantumCircuit(10)
        qc.append(perm, [0, 2, 4, 6, 8])

        coupling_map = CouplingMap.from_ring(10)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        qc_transpiled = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        ).run(qc)
        self.assertEqual(qc_transpiled, qc)

    def test_abstract_synthesis_all_permutations(self):
        """Test abstract synthesis of permutation gates, varying permutation gate patterns."""

        edges = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]

        coupling_map = CouplingMap()
        for i in range(5):
            coupling_map.add_physical_qubit(i)
        for edge in edges:
            coupling_map.add_edge(*edge)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        pm = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=False
            )
        )

        for pattern in itertools.permutations(range(4)):
            qc = QuantumCircuit(5)
            qc.append(PermutationGate(pattern), [2, 0, 3, 1])
            self.assertIn("permutation", qc.count_ops())

            qc_transpiled = pm.run(qc)
            self.assertNotIn("permutation", qc_transpiled.count_ops())

            self.assertEqual(Operator(qc), Operator(qc_transpiled))

    def test_concrete_synthesis_all_permutations(self):
        """Test concrete synthesis of permutation gates, varying permutation gate patterns."""

        edges = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]

        coupling_map = CouplingMap()
        for i in range(5):
            coupling_map.add_physical_qubit(i)
        for edge in edges:
            coupling_map.add_edge(*edge)

        synthesis_config = HLSConfig(permutation=[("token_swapper", {"trials": 10})])
        pm = PassManager(
            HighLevelSynthesis(
                synthesis_config, coupling_map=coupling_map, target=None, use_qubit_indices=True
            )
        )

        for pattern in itertools.permutations(range(4)):

            qc = QuantumCircuit(5)
            qc.append(PermutationGate(pattern), [2, 0, 3, 1])
            self.assertIn("permutation", qc.count_ops())

            qc_transpiled = pm.run(qc)
            self.assertNotIn("permutation", qc_transpiled.count_ops())
            self.assertEqual(Operator(qc), Operator(qc_transpiled))

            for inst in qc_transpiled:
                qubits = tuple(q.index for q in inst.qubits)
                self.assertIn(qubits, edges)


if __name__ == "__main__":
    unittest.main()
