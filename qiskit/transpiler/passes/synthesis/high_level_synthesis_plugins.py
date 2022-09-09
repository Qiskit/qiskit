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
Default plugins for synthesizing high-level-objects in Qiskit.
"""

import stevedore

from qiskit.quantum_info import decompose_clifford
from qiskit.transpiler.synthesis import cnot_synth


class HighLevelSynthesisPluginManager:
    """Class tracking the installed high-level-synthesis plugins."""

    def __init__(self):
        self.plugins = stevedore.ExtensionManager(
            "qiskit.synthesis", invoke_on_load=True, propagate_map_exceptions=True
        )

        # The registered plugin names should be of the form <OperationName.SynthesisMethodName>.

        # Create a dict, mapping <OperationName> to the list of its <SynthesisMethodName>s.
        self.plugins_by_op = dict()
        for plugin_name in self.plugins.names():
            op_name, method_name = plugin_name.split(".")
            if op_name not in self.plugins_by_op.keys():
                self.plugins_by_op[op_name] = []
            self.plugins_by_op[op_name].append(method_name)

    def method_names(self, op_name):
        """Returns plugin methods for op_name."""
        if op_name in self.plugins_by_op.keys():
            return self.plugins_by_op[op_name]
        else:
            return []

    def method(self, op_name, method_name):
        """Returns the plugin for ``op_name`` and ``method_name``."""
        plugin_name = op_name + "." + method_name
        return self.plugins[plugin_name].obj


class DefaultSynthesisClifford:
    def run(self, clifford, **options):
        """Run synthesis for the given Clifford."""

        print(f"    -> Running DefaultSynthesisClifford")
        decomposition = decompose_clifford(clifford)
        return decomposition


class DefaultSynthesisLinearFunction:
    def run(self, linear_function, **options):
        """Run synthesis for the given LinearFunction."""

        print(f"    -> Running DefaultSynthesisLinearFunction")
        decomposition = cnot_synth(linear_function.linear)
        return decomposition
