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

from qiskit.circuit import QuantumCircuit, Operation
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig


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


if __name__ == "__main__":
    unittest.main()
