# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests preset pass manager API"""

import unittest
from ddt import ddt, data

from qiskit import QuantumCircuit, default_passmanager
from qiskit.transpiler import CouplingMap, PassManager, Target
from qiskit.circuit.library import XGate
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    RemoveResetInZeroState,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import level0, level1, level2, level3
from qiskit.transpiler.preset_passmanagers.builtin_plugins import OptimizationPassManager
from test import QiskitTestCase  # pylint: disable=wrong-import-order

from ..legacy_cmaps import LAGOS_CMAP


def mock_get_passmanager_stage(
    stage_name,
    plugin_name,
    pm_config,
    optimization_level=None,  # pylint: disable=unused-argument
) -> PassManager:
    """Mock function for get_passmanager_stage."""
    if stage_name == "translation" and plugin_name == "custom_stage_for_test":
        pm = PassManager([RemoveResetInZeroState()])
        return pm

    elif stage_name == "scheduling" and plugin_name == "custom_stage_for_test":
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(pm_config.instruction_durations),
                PadDynamicalDecoupling(pm_config.instruction_durations, dd_sequence),
            ]
        )
        return pm
    elif stage_name == "init":
        return PassManager([])
    elif stage_name == "routing":
        return PassManager([])
    elif stage_name == "optimization":
        return OptimizationPassManager().pass_manager(pm_config, optimization_level)
    elif stage_name == "layout":
        return PassManager([])
    else:
        raise Exception("Failure, unexpected stage plugin combo for test")


@ddt
class TestDefaultPassManager(QiskitTestCase):
    """Test default_passmanager function."""

    @data(0, 1, 2, 3)
    def test_with_target(self, optimization_level):
        """Test a passmanager is constructed from a target."""
        target = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP).target
        pm = default_passmanager(target, optimization_level=optimization_level)
        self.assertIsInstance(pm, PassManager)

    def test_invalid_optimization_level(self):
        """Assert we fail with an invalid optimization_level."""
        with self.assertRaises(ValueError):
            default_passmanager(optimization_level=42)

    @unittest.mock.patch.object(
        level2.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level2(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 2

        class ABackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = ABackend(num_qubits=7, coupling_map=LAGOS_CMAP).target
        pm = default_passmanager(target=target, optimization_level=optimization_level)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [
            x.__class__.__name__ for x in pm.translation.to_flow_controller().tasks
        ]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    @unittest.mock.patch.object(
        level1.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level1(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 1

        class ABackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = ABackend(num_qubits=7, coupling_map=LAGOS_CMAP).target
        pm = default_passmanager(target=target, optimization_level=optimization_level)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [
            x.__class__.__name__ for x in pm.translation.to_flow_controller().tasks
        ]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    @unittest.mock.patch.object(
        level3.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level3(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 3

        class ABackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = ABackend(num_qubits=7, coupling_map=LAGOS_CMAP).target
        pm = default_passmanager(target=target, optimization_level=optimization_level)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [
            x.__class__.__name__ for x in pm.translation.to_flow_controller().tasks
        ]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    @unittest.mock.patch.object(
        level0.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level0(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 0

        class ABackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = ABackend(num_qubits=7, coupling_map=LAGOS_CMAP).target
        pm = default_passmanager(target, optimization_level=optimization_level)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    def test_generate_preset_pass_manager_with_list_coupling_map(self):
        """Test that generate_preset_pass_manager can handle list-based coupling_map."""

        # Define the coupling map as a list
        coupling_map_list = [[0, 1]]
        coupling_map_object = CouplingMap(coupling_map_list)

        # Circuit that doesn't fit in the coupling map
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.measure_all()

        target = Target.from_configuration(
            ["cx", "u"], num_qubits=2, coupling_map=coupling_map_object
        )

        pm_list = default_passmanager(target, optimization_level=0, seed_transpiler=42)
        pm_object = default_passmanager(target, optimization_level=0, seed_transpiler=42)

        transpiled_circuit_list = pm_list.run(qc)
        transpiled_circuit_object = pm_object.run(qc)

        # Check if both are instances of PassManager
        self.assertIsInstance(pm_list, PassManager)
        self.assertIsInstance(pm_object, PassManager)

        # Ensure the DAGs from both methods are identical
        self.assertEqual(transpiled_circuit_list, transpiled_circuit_object)
