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


from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from .high_level_synthesis_plugins import HighLevelSynthesisPluginManager


class HLSConfig:
    """
    For each higher-level-object (e.g., "clifford", "linear_function", etc.) the
    high-level-synthesis config allows to specify a list of "methods", each method
    represented by a pair consisting of a name of the synthesis algorithm and of a dictionary
    providing additional arguments for this algorithm.

    The names of the synthesis algorithms should be declared in ``entry_points`` for
    ``qiskit.synthesis`` in ``setup.py``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.
    """

    def __init__(self, use_default_on_unspecified=True, **kwargs):
        """
        Creates a high-level-synthesis config.
        Args:
            use_default_on_unspecified (bool): if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.methods = dict()

        for key, value in kwargs.items():
            print("%s == %s" % (key, value))
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods

    def print(self):
        """This is temporary for debugging."""
        print("HLS CONFIG:")
        print(f"use_default_on_unspecified = {self.use_default_on_unspecified}")
        for hls_name in self.methods:
            print(f"  name = {hls_name}, method = {self.methods[hls_name]}")


# ToDo: Do we have a way to specify optimization criteria (e.g., 2q gate count vs. depth)?


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

        Raises:
            TranspilerError: when the specified synthesis method is not available.
        """

        hls_plugin_manager = HighLevelSynthesisPluginManager()

        for node in dag.op_nodes():

            if node.name in self.hls_config.methods.keys():
                # the operation's name appears in the user-provided config,
                # we use the list of methods provided by the user
                methods = self.hls_config.methods[node.name]
            elif (
                self.hls_config.use_default_on_unspecified
                and "default" in hls_plugin_manager.method_names(node.name)
            ):
                # the operation's name does not appear in the user-specified config,
                # we use the "default" method when instructed to do so and the "default"
                # method is available
                methods = [("default", {})]
            else:
                methods = []

            for method in methods:
                plugin_name, plugin_args = method

                if plugin_name not in hls_plugin_manager.method_names(node.name):
                    raise TranspilerError(
                        "Specified method: %s not found in available plugins for %s"
                        % (plugin_name, node.name)
                    )

                print(f"  Using method {plugin_name} for {node.name}")
                plugin_method = hls_plugin_manager.method(node.name, plugin_name)

                # ToDo: similarly to UnitarySynthesis, we should pass additional parameters
                #       e.g. coupling_map to the synthesis algorithm.
                decomposition = plugin_method.run(node.op, **plugin_args)

                # The synthesis methods that are not suited for the given higher-level-object
                # will return None, in which case the next method in the list will be used.
                if decomposition is not None:
                    dag.substitute_node_with_dag(node, circuit_to_dag(decomposition))
                    break

        return dag
