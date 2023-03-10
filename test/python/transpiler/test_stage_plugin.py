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
Tests for the staged transpiler plugins.
"""

from test import combine

import ddt

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.compiler.transpiler import transpile
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager, PassManagerConfig, CouplingMap
from qiskit.transpiler.preset_passmanagers.builtin_plugins import BasicSwapPassManager
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
    list_stage_plugins,
    passmanager_stage_plugins,
)
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.basicaer import QasmSimulatorPy


class TestStagePassManagerPlugin(QiskitTestCase):
    """Tests for the transpiler stage plugin interface."""

    def test_list_stage_plugins(self):
        """Test list stage plugin function."""
        routing_passes = list_stage_plugins("routing")
        self.assertIn("basic", routing_passes)
        self.assertIn("sabre", routing_passes)
        self.assertIn("lookahead", routing_passes)
        self.assertIn("stochastic", routing_passes)
        self.assertIsInstance(list_stage_plugins("init"), list)
        self.assertIsInstance(list_stage_plugins("layout"), list)
        self.assertIsInstance(list_stage_plugins("translation"), list)
        self.assertIsInstance(list_stage_plugins("optimization"), list)
        self.assertIsInstance(list_stage_plugins("scheduling"), list)

    def test_list_stage_plugins_invalid_stage_name(self):
        """Test list stage plugin function with invalid stage name."""
        with self.assertRaises(TranspilerError):
            list_stage_plugins("not_a_stage")

    def test_passmanager_stage_plugins(self):
        """Test entry_point_obj function."""
        basic_obj = passmanager_stage_plugins("routing")
        self.assertIsInstance(basic_obj["basic"], BasicSwapPassManager)

    def test_passmanager_stage_plugins_not_found(self):
        """Test entry_point_obj function with nonexistent stage"""
        with self.assertRaises(TranspilerError):
            passmanager_stage_plugins("foo_stage")

    def test_build_pm_invalid_plugin_name_valid_stage(self):
        """Test get pm from plugin with invalid plugin name and valid stage."""
        plugin_manager = PassManagerStagePluginManager()
        with self.assertRaises(TranspilerError):
            plugin_manager.get_passmanager_stage("init", "empty_plugin", PassManagerConfig())

    def test_build_pm_invalid_stage(self):
        """Test get pm from plugin with invalid stage."""
        plugin_manager = PassManagerStagePluginManager()
        with self.assertRaises(TranspilerError):
            plugin_manager.get_passmanager_stage(
                "not_a_sage", "fake_plugin_not_real", PassManagerConfig()
            )

    def test_build_pm(self):
        """Test get pm from plugin."""
        plugin_manager = PassManagerStagePluginManager()
        pm_config = PassManagerConfig()
        pm = plugin_manager.get_passmanager_stage(
            "routing", "sabre", pm_config, optimization_level=3
        )
        self.assertIsInstance(pm, PassManager)


@ddt.ddt
class TestBuiltinPlugins(QiskitTestCase):
    """Test that all built-in plugins work in transpile()."""

    @combine(
        optimization_level=list(range(4)),
        routing_method=["basic", "lookahead", "sabre", "stochastic"],
    )
    def test_routing_plugins(self, optimization_level, routing_method):
        """Test all routing plugins (excluding error)."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.measure_all()
        tqc = transpile(
            qc,
            basis_gates=["cx", "sx", "x", "rz"],
            coupling_map=CouplingMap.from_line(4),
            optimization_level=optimization_level,
            routing_method=routing_method,
        )
        backend = QasmSimulatorPy()
        counts = backend.run(tqc, shots=1000).result().get_counts()
        self.assertDictAlmostEqual(counts, {"0000": 500, "1111": 500}, delta=100)
