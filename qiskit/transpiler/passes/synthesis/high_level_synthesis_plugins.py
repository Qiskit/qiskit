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

import abc

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


class HighLevelSynthesisPlugin(abc.ABC):
    """Abstract high-level synthesis plugin class.

    This abstract class defines the interface for high-level synthesis plugins.
    """

    @abc.abstractmethod
    def run(self, high_level_object, **options):
        """Run synthesis for the given Operation.

        Args:
            high_level_object (Operation): The Operation to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object
            options: The optional kwargs.

        Returns:
            QuantumCircuit: The quantum circuit representation of the Operation
                when successful, and ``None`` otherwise.
        """
        pass


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given Clifford."""
        decomposition = decompose_clifford(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""
        decomposition = cnot_synth(high_level_object.linear)
        return decomposition
