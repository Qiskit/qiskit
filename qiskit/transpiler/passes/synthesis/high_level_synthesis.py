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


"""Synthesize higher-level objects."""

import stevedore

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError


# ToDo: possibly choose a better name for this data structure.
class HLSConfig:
    """
    To each higher-level-object (e.g., "clifford", "linear_function", etc.) we associate a
    pair consisting of the name of a synthesis algorithm and additional arguments it requires.
    There are two special values for the name:
        "default" is the default synthesis algorithm (chosen by Qiskit developers)
        "none" is to skip synthesizing this object
    All other names should be declared in entry points in setup.
    """

    def __init__(self, default_or_none: True):
        """
        Creates a high-level-synthesis config.
        Args:
            default_or_none: specifies the default method for every higher-level object,
                either "default" (use the "default" synthesize method if available),
                or "none" (skip synthesizing)
        """
        self.default_or_none = default_or_none
        self.methods = dict()

    def set_method(self, hls_name: str, method_name: str, method_args=None):
        """Sets the synthesis method for a given higher-level-object."""
        if method_args is None:
            method_args = dict()
        self.methods[hls_name] = (method_name, method_args)

    def print(self):
        """This is temporary for debugging."""
        print("HLS CONFIG:")
        print(f"default_or_none = {self.default_or_none}")
        for hls_name in self.methods:
            print(f"  name = {hls_name}, method = {self.methods[hls_name]}")


# ToDo [1]: Make sure that plugin_method is an instance of a certain API
#           (for instance, every plugin for Unitary Synthesis derives from UnitarySynthesisPlugin).
#           It probably makes sense to create a more general HigherLevelSynthesisPlugin
#           (which is exactly what UnitarySynthesisPlugin is right now), and to rename/inherit
#           UnitarySynthesisPlugin from that.
# ToDo [2]: Do we have a way to specify optimization criteria (e.g., 2q gate count vs. depth)?
# ToDo [3]: Do we also need a PluginManager, similar to UnitarySynthesisPluginManager?
class HighLevelSynthesis(TransformationPass):
    """Synthesize higher-level objects by choosing the appropriate synthesis method
    based on the node's name and the HLS-config (if provided).
    """

    def __init__(self, hls_config=None):
        super().__init__()

        if hls_config is not None:
            self.hls_config = hls_config
        else:
            # When the config file is not provided, we will use the "default" method
            # to synthesize Operations (when available).
            self.hls_config = HLSConfig(True)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with certain Operations synthesized (as specified by
            the hls_config).
        """

        for node in dag.op_nodes():
            # Get available plugins corresponding to the name of the current operation (node)
            # ToDo [4]: check how (un)efficient it is to call stevedore for every gate in the circuit
            available_plugins = stevedore.ExtensionManager(
                "qiskit.synthesis." + node.name, invoke_on_load=True, propagate_map_exceptions=True
            )

            if node.name in self.hls_config.methods.keys():
                # the operation's name appears in the user-provided config
                # note that the plugin might be "default" or "none"
                plugin_name, plugin_args = self.hls_config.methods[node.name]
            elif self.hls_config.default_or_none and "default" in available_plugins.names():
                # the operation's name does not appear in the user-specified config,
                # but we should use the "default" method when possible
                plugin_name, plugin_args = "default", {}
            else:
                # the operation's name does not appear in the user-specified config,
                # and either the "default" method is not available or we should skip these
                plugin_name, plugin_args = "none", {}

            if plugin_name == "none":
                # don't synthesize this operation
                continue

            if plugin_name not in available_plugins:
                raise TranspilerError(
                    "Specified method: %s not found in available plugins for %s"
                    % (plugin_name, node.name)
                )

            # print(f"  Using method {plugin_name} for {node.name}")
            plugin_method = available_plugins[plugin_name].obj

            # ToDo: [5] similarly to UnitarySynthesis, we should pass additional parameters
            #       e.g. coupling_map to the synthesis algorithm.
            decomposition = plugin_method.run(node.op, **plugin_args)
            dag.substitute_node_with_dag(node, circuit_to_dag(decomposition))

        return dag
