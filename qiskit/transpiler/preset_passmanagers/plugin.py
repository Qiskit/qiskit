# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================================================================
Transpiler Stage Plugin Interface (:mod:`qiskit.transpiler.preset_passmanagers.plugin`)
=======================================================================================

.. currentmodule:: qiskit.transpiler.preset_pass_managers.plugin

This module defines the plugin interface for providing custom stage
implementations for the preset pass managers and the :func:`~.transpile`
function. This enables external Python packages to provide
:class:`~.PassManager` objects that can be used for each stage.

.. _stage_table:

Plugin Stages
=============

Currently there are 6 stages in the preset pass managers used by and corresponding entrypoints.

.. list-table:: Stages
   :header-rows: 1

   * - Stage Name
     - Entry Point
     - Reserved Names
     - Description and expectations
   * - ``init``
     - ``qiskit.transpiler.init``
     - No reserved names
     - This stage runs first and is typically used for an initial logical optimization also
       for most ``layout`` and ``routing`` stages this stage is used to translate any gates
       that operate on more than 2 qubits into gates that operate on 1 or 2
       qubits only. This is because most layout and routing algorithms are
       only designed to work with 1 and 2 qubit gates.
       ``init``
   * - ``layout``
     - ``qiskit.transpiler.layout``
     - ``trivial``, ``dense``, ``noise_adaptive``, ``sabre``
     - The output from this staged is expected to have the ``layout`` property
       set field set with a :class:`~.Layout` object. Additionally, the circuit is
       is typically expected to be embedded so that it expanded to include all
       qubits and the :class:`~.ApplyLayout` pass is expected to be run to apply the
       layout. The embedding of the :class:`~.Layout` can be generated with
       :func:`~.generate_embed_passmanager`.
   * - ``routing``
     - ``qiskit.transpiler.routing``
     - ``basic``, ``stochastic``, ``lookahead``, ``sabre``, ``toqm``
     - The output from this stage is expected to have the circuit match the
       connectivity constraints of the target backend. This does not necessarily
       need to match the directionality of the edges in the target as a later
       stage typically will adjust directional gates to match that constraint
       (but there is no penalty for doing that in the ``routing`` stage).
   * - ``translation``
     - ``qiskit.transpiler.translation``
     - ``translator``, ``synthesis``, ``unroller``
     - The output of this stage is expected to have every operation be a native
        instruction on the target backend.
   * - ``optimization``
     - ``qiskit.transpiler.optimization``
     - There are no reserve plugin names
     - This stage is expected to perform and optimization and simplification.
       The constraints from earlier stages still apply to the output of this
       stage. After the ``optimization`` stage is run we expect the circuit
       to still be executable on the target.
   * - ``scheduling``
     - ``qiskit.transpiler.scheduling``
     - ``alap``, ``asap``
     - This is the last stage run and it is expected to output a scheduled
       circuit such that all idle periods in the circuit are marked by explicit
       :class:`~.Delay` instructions.

Writing Plugins
===============

To write a pass manager stage plugin there are 2 main steps. The first step is
to create a subclass of the abstract plugin class
:class:`~.PassManagerStagePluginManager` which is used to define how the :class:`~.PassManager`
for the stage will be constructed. For example, to create a ``layout`` stage plugin that just
runs :class:`~.VF2Layout`::

    from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
    from qiskit.transpiler.preset_passmanagers import common
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import VF2Layout

    class VF2LayoutPlugin(PassManagerStagePlugin):

        def pass_manager(self, pass_manager_config):
            layout_pm = PassManager([VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                properties=pass_manager_config.backend_properties,
                target=pass_manager_config.target
            )])
            layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
            return layout_pm

The second step is to expose the :class:`~.PassManagerStagePluginManager`
subclass as a setuptools entry point in the package metadata. This can be done
by simply adding an ``entry_points`` entry to the ``setuptools.setup`` call in
the ``setup.py`` or the plugin package with the necessary entry points under the
appropriate namespace for the stage your plugin is for. You can see the list
of stages, entrypoints, and expectations from the stage in :ref:`stage_table`.
For example, continuing from the example plugin above::

    entry_points = {
        'qiskit.transpiler.layout': [
            'vf2 = qiskit_plugin_pkg.module.plugin:VF2LayoutPlugin',
        ]
    },

(note that the entry point ``name = path`` is a single string not a Python
expression). There isn't a limit to the number of plugins a single package can
include as long as each plugin has a unique name. So a single package can
expose multiple plugins if necessary. Refer to :ref:`stage_table` for a list
of reserved names for each stage.

Plugin API
==========

.. autosummary::
   :toctree: ../stubs/

   PassManagerStagePlugin
   PassManagerStagePluginManager
   list_stage_plugins
"""

import abc
from typing import List

import stevedore

from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager_config import PassManagerConfig


class PassManagerStagePlugin(abc.ABC):
    """A ``PassManagerStagePlugin`` is a plugin interface object for using custom
    stages in :func:`transpile()`.

    A ``PassManagerStagePlugin`` object can be added to an external package and

    """

    @abc.abstractmethod
    def pass_manager(self, pass_manager_config: PassManagerConfig) -> PassManager:
        """This method is designed to return a :class:`~.PassManager` for the"""
        pass


class PassManagerStagePluginManager:
    """Manager class for preset pass manager stage plugins."""

    def __init__(self):
        super().__init__()
        self.init_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.init", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.layout_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.layout", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.routing_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.routing", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.translation_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.translation", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.optimization_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.optimization", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.scheduling_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.scheduling", invoke_on_load=True, propagate_map_exceptions=True
        )

    def get_passmanager_stage(
        self, stage_name: str, plugin_name: str, pm_config: PassManagerConfig
    ) -> PassManager:
        """Get a stage"""
        if stage_name == "init":
            return self._build_pm(self.init_plugins, stage_name, plugin_name, pm_config)
        elif stage_name == "layout":
            return self._build_pm(self.layout_plugins, stage_name, plugin_name, pm_config)
        elif stage_name == "routing":
            return self._build_pm(self.routing_plugins, stage_name, plugin_name, pm_config)
        elif stage_name == "translation":
            return self._build_pm(self.translation_plugins, stage_name, plugin_name, pm_config)
        elif stage_name == "optimization_plugins":
            return self._build_pm(self.optimization_plugins, stage_name, plugin_name, pm_config)
        elif stage_name == "scheduling":
            return self._build_pm(self.scheduling_plugins, stage_name, plugin_name, pm_config)
        else:
            raise TranspilerError(f"Invalid stage name: {stage_name}")

    def _build_pm(
        self,
        stage_obj: stevedore.ExtensionManager,
        stage_name: str,
        plugin_name: str,
        pm_config: PassManagerConfig,
    ):
        plugin_obj = stage_obj.get(plugin_name)
        if plugin_obj is None:
            raise TranspilerError(f"Invalid plugin name {plugin_name} for stage {stage_name}")
        return plugin_obj.pass_manager(pm_config)


def list_stage_plugins(stage_name: str) -> List[str]:
    """Get a list of installed plugins for a stage.

    Args:
        stage_name: The stage name to get the plugin names for

    Returns:
        plugins: The list of installed plugin names for the specified stages

    Raises:
       TranspilerError: If an invalid stage name is specified.
    """
    plugin_mgr = PassManagerStagePluginManager()
    if stage_name == "init":
        return plugin_mgr.init_plugins.names()
    elif stage_name == "layout":
        return plugin_mgr.layout_plugins.names()
    elif stage_name == "routing":
        return plugin_mgr.routing_plugins.names()
    elif stage_name == "translation":
        return plugin_mgr.translation_plugins.names()
    elif stage_name == "optimization_plugins":
        return plugin_mgr.optimization_plugins.names()
    elif stage_name == "scheduling":
        return plugin_mgr.scheduling_plugins.names()
    else:
        raise TranspilerError(f"Invalid stage name: {stage_name}")
