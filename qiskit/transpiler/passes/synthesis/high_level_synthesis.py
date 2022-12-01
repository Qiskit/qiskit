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
from qiskit.quantum_info import decompose_clifford
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin
from qiskit.transpiler import CouplingMap
import rustworkx as rx

class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level-objects. A higher-level object is an object of type
    :class:`~.Operation` (e.g., "clifford", "linear_function", etc.), and the list
    of applicable synthesis methods is strictly tied to the name of the operation.
    In the config, each method is represented by a pair consisting of a name of the synthesis
    algorithm and of a dictionary providing additional arguments for this algorithm.

    The names of the synthesis algorithms should be declared in ``entry_points`` for
    ``qiskit.synthesis`` in ``setup.py``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of creating and using such config files, refer to the
    documentation for :class:`~.HighLevelSynthesis`.
    """

    def __init__(self, use_default_on_unspecified=True, **kwargs):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified (bool): if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.methods = dict()

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


# ToDo: Do we have a way to specify optimization criteria (e.g., 2q gate count vs. depth)?


class HighLevelSynthesis(TransformationPass):
    """Synthesize higher-level objects by choosing the appropriate synthesis method
    based on the object's name and the high-level-synthesis config of type
    :class:`~.HLSConfig` (if provided).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.
    """

    def __init__(
        self,
        coupling_map: CouplingMap=None,
        hls_config=None
    ):
        """
        HighLevelSynthesis initializer.

        Args:
            coupling_map (CouplingMap): the coupling map of the backend
                in case synthesis is done on a physical circuit.
            hls_config (HLSConfig): the high-level-synthesis config file
            specifying synthesis methods and parameters.
        """
        super().__init__()

        print(f"HLS::init {coupling_map = }")
        self._coupling_map = coupling_map

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
        print(f"HLS::run {self._coupling_map = }")
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

                plugin_method = hls_plugin_manager.method(node.name, plugin_name)

                if self._coupling_map:
                    plugin_args["coupling_map"] = self._coupling_map

                decomposition = plugin_method.run(node.op, **plugin_args)

                # The synthesis methods that are not suited for the given higher-level-object
                # will return None, in which case the next method in the list will be used.
                if decomposition is not None:
                    dag.substitute_node_with_dag(node, circuit_to_dag(decomposition))
                    break

        return dag


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given Clifford."""
        print(f"DefaultSynthesisClifford {options = }")

        decomposition = decompose_clifford(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""
        print(f"DefaultSynthesisLinearFunction {options = }")

        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""
        print(f"KMSSynthesisLinearFunction {options = }")
        coupling_map = options.get("coupling_map", None)

        if coupling_map:
            longest_path = _get_longest_line(coupling_map)
        else:
            longest_path = list(range())

            # The KMS algorithm
            if len(longest_path) < len(high_level_object.linear):
                return None

        print(f"{coupling_map = }")
        print(f"{longest_path = }")

        decomposition = synth_cnot_depth_line_kms(high_level_object.linear)
        return decomposition


class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""
        print(f"PMHSynthesisLinearFunction {options = }")

        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


def _get_longest_line(coupling_map: CouplingMap) -> list:
    """Gets the longest line from the coupling map."""
    graph = coupling_map.graph
    # longest_path = rx.longest_simple_path(graph)
    simple_paths_generator = (y.values() for y in rx.all_pairs_all_simple_paths(graph).values())
    all_simple_paths = [x[0] for y in simple_paths_generator for x in y]
    longest_path = max(all_simple_paths, key=len)
    return list(longest_path)
