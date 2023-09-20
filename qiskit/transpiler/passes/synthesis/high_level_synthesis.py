# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Synthesize higher-level objects."""

from typing import Optional
import rustworkx as rx

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper

from qiskit.synthesis.clifford import (
    synth_clifford_full,
    synth_clifford_layers,
    synth_clifford_depth_lnn,
    synth_clifford_greedy,
    synth_clifford_ag,
    synth_clifford_bm,
)
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.permutation import (
    synth_permutation_basic,
    synth_permutation_acg,
    synth_permutation_depth_lnn_kms,
)

from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin


class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level-objects. A higher-level object is an object of type
    :class:`~.Operation` (e.g., "clifford", "linear_function", etc.), and the list
    of applicable synthesis methods is strictly tied to the name of the operation.
    In the config, each method is specified as a tuple consisting of the name of the
    synthesis algorithm and of a dictionary providing additional arguments for this
    algorithm. Additionally, a synthesis method can be specified as a tuple consisting
    of an instance of :class:`.HighLevelSynthesisPlugin` and additional arguments.
    Moreover, when there are no additional arguments, a synthesis
    method can be specified simply by name or by an instance
    of :class:`.HighLevelSynthesisPlugin`. The following example illustrates different
    ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis algorithms should be declared in ``entry_points`` for
    ``qiskit.synthesis`` in ``setup.py``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the
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
        self.methods = {}

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


# ToDo: Do we have a way to specify optimization criteria (e.g., 2q gate count vs. depth)?


class HighLevelSynthesis(TransformationPass):
    """Synthesize higher-level objects using synthesis plugins.

    Synthesis plugins apply synthesis methods specified in the high-level-synthesis
    config (refer to the documentation for :class:`~.HLSConfig`).

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
        hls_config: Optional[HLSConfig] = None,
        coupling_map: Optional[CouplingMap] = None,
        target: Optional[Target] = None,
        use_qubit_indices: bool = False,
    ):
        """
        HighLevelSynthesis initializer.

        Args:
            hls_config: Optional, the high-level-synthesis config that specifies synthesis methods
                and parameters for various high-level-objects in the circuit. If it is not specified,
                the default synthesis methods and parameters will be used.
            coupling_map: Optional, directed graph represented as a coupling map.
            target: Optional, the backend target to use for this pass. If it is specified,
                it will be used instead of the coupling map.
            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after
                the layout is set, that is, whether the qubit indices of higher-level-objects correspond
                to qubit indices on the target backend.
        """
        super().__init__()

        if hls_config is not None:
            self.hls_config = hls_config
        else:
            # When the config file is not provided, we will use the "default" method
            # to synthesize Operations (when available).
            self.hls_config = HLSConfig(True)
        self.hls_plugin_manager = HighLevelSynthesisPluginManager()
        self._coupling_map = coupling_map
        self._target = target
        self._use_qubit_indices = use_qubit_indices
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()

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
        # If there aren't any high level operations to synthesize return fast
        hls_names = set(self.hls_plugin_manager.plugins_by_op)
        if not hls_names.intersection(dag.count_ops()):
            return dag
        for node in dag.op_nodes():
            if node.name in self.hls_config.methods.keys():
                # the operation's name appears in the user-provided config,
                # we use the list of methods provided by the user
                methods = self.hls_config.methods[node.name]
            elif (
                self.hls_config.use_default_on_unspecified
                and "default" in self.hls_plugin_manager.method_names(node.name)
            ):
                # the operation's name does not appear in the user-specified config,
                # we use the "default" method when instructed to do so and the "default"
                # method is available
                methods = ["default"]
            else:
                methods = []

            for method in methods:
                # There are two ways to specify a synthesis method. The more explicit
                # way is to specify it as a tuple consisting of a synthesis algorithm and a
                # list of additional arguments, e.g.,
                #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}), or
                #   ("pmh", {}).
                # When the list of additional arguments is empty, one can also specify
                # just the synthesis algorithm, e.g.,
                #   "pmh".
                if isinstance(method, tuple):
                    plugin_specifier, plugin_args = method
                else:
                    plugin_specifier = method
                    plugin_args = {}

                # There are two ways to specify a synthesis algorithm being run,
                # either by name, e.g. "kms" (which then should be specified in entry_points),
                # or directly as a class inherited from HighLevelSynthesisPlugin (which then
                # does not need to be specified in entry_points).
                if isinstance(plugin_specifier, str):
                    if plugin_specifier not in self.hls_plugin_manager.method_names(node.name):
                        raise TranspilerError(
                            "Specified method: %s not found in available plugins for %s"
                            % (plugin_specifier, node.name)
                        )
                    plugin_method = self.hls_plugin_manager.method(node.name, plugin_specifier)
                else:
                    plugin_method = plugin_specifier

                qubits = (
                    [dag.find_bit(x).index for x in node.qargs] if self._use_qubit_indices else None
                )

                decomposition = plugin_method.run(
                    node.op,
                    coupling_map=self._coupling_map,
                    target=self._target,
                    qubits=qubits,
                    **plugin_args,
                )

                # The synthesis methods that are not suited for the given higher-level-object
                # will return None, in which case the next method in the list will be used.
                if decomposition is not None:
                    dag.substitute_node_with_dag(node, circuit_to_dag(decomposition))
                    break

        return dag


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin.

    For N <= 3 qubits this is the optimal CX cost decomposition by Bravyi, Maslov.
    For N > 3 qubits this is done using the general non-optimal greedy compilation
    routine from reference by Bravyi, Hu, Maslov, Shaydulin.

    This plugin name is :``clifford.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_full(high_level_object)
        return decomposition


class AGSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Aaronson-Gottesman method.

    This plugin name is :``clifford.ag`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_ag(high_level_object)
        return decomposition


class BMSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method.
    The plugin is named

    The method only works on Cliffords with at most 3 qubits, for which it
    constructs the optimal CX cost decomposition.

    This plugin name is :``clifford.bm`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if high_level_object.num_qubits <= 3:
            decomposition = synth_clifford_bm(high_level_object)
        else:
            decomposition = None
        return decomposition


class GreedySynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the greedy synthesis
    Bravyi-Hu-Maslov-Shaydulin method.

    This plugin name is :``clifford.greedy`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_greedy(high_level_object)
        return decomposition


class LayerSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers.

    This plugin name is :``clifford.layers`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_layers(high_level_object)
        return decomposition


class LayerLnnSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers, with each layer synthesized
    adhering to LNN connectivity.

    This plugin name is :``clifford.lnn`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_depth_lnn(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin.

    This plugin name is :``linear_function.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method.

    This plugin name is :``linear_function.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        decomposition = synth_cnot_depth_line_kms(high_level_object.linear)
        return decomposition


class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method.

    This plugin name is :``linear_function.pmh`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method.

    This plugin name is :``permutation.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin name is :``permutation.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin name is :``permutation.acg`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition


class TokenSwapperSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the token swapper algorithm.

    This plugin name is :``permutation.token_swapper`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    In more detail, this plugin is used to synthesize objects of type `PermutationGate`.
    When synthesis succeeds, the plugin outputs a quantum circuit consisting only of swap
    gates. When synthesis does not succeed, the plugin outputs `None`.

    If either `coupling_map` or `qubits` is None, then the synthesized circuit
    is not required to adhere to connectivity constraints, as is the case
    when the synthesis is done before layout/routing.

    On the other hand, if both `coupling_map` and `qubits` are specified, the synthesized
    circuit is supposed to adhere to connectivity constraints. At the moment, the
    plugin only creates swap gates between qubits in `qubits`, i.e. it does not use
    any other qubits in the coupling map (if such synthesis is not possible, the
    plugin  outputs `None`).

    The plugin supports the following plugin-specific options:

    * trials: The number of trials for the token swapper to perform the mapping. The
      circuit with the smallest number of SWAPs is returned.
    * seed: The argument to the token swapper specifying the seed for random trials.
    * parallel_threshold: The argument to the token swapper specifying the number of nodes
      in the graph beyond which the algorithm will use parallel processing.

    For more details on the token swapper algorithm, see to the paper:
    `arXiv:1809.03452 <https://arxiv.org/abs/1809.03452>`__.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""

        trials = options.get("trials", 5)
        seed = options.get("seed", 0)
        parallel_threshold = options.get("parallel_threshold", 50)

        pattern = high_level_object.pattern
        pattern_as_dict = {j: i for i, j in enumerate(pattern)}

        # When the plugin is called from the HighLevelSynthesis transpiler pass,
        # the coupling map already takes target into account.
        if coupling_map is None or qubits is None:
            # The abstract synthesis uses a fully connected coupling map, allowing
            # arbitrary connections between qubits.
            used_coupling_map = CouplingMap.from_full(len(pattern))
        else:
            # The concrete synthesis uses the coupling map restricted to the set of
            # qubits over which the permutation gate is defined. If we allow using other
            # qubits in the coupling map, replacing the node in the DAGCircuit that
            # defines this PermutationGate by the DAG corresponding to the constructed
            # decomposition becomes problematic. Note that we allow the reduced
            # coupling map to be disconnected.
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        graph = rx.PyGraph()
        graph.extend_from_edge_list(list(used_coupling_map.get_edges()))
        swapper = ApproximateTokenSwapper(graph, seed=seed)

        try:
            swapper_result = swapper.map(
                pattern_as_dict, trials, parallel_threshold=parallel_threshold
            )
        except rx.InvalidMapping:
            swapper_result = None

        if swapper_result is not None:
            decomposition = QuantumCircuit(len(graph.node_indices()))
            for swap in swapper_result:
                decomposition.swap(*swap)
            return decomposition

        return None
