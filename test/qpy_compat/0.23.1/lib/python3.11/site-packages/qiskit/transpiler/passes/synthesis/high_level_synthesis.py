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
from qiskit.synthesis import synth_permutation_basic, synth_permutation_acg
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.synthesis import synth_clifford_full
from qiskit.synthesis.linear import synth_cnot_count_full_pmh
from qiskit.synthesis.permutation import synth_permutation_depth_lnn_kms
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin


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


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given Clifford."""
        decomposition = synth_clifford_full(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""
        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method.


    This plugin can be accessed by the ``kms`` method name in the
    ``HLSConfig`` for ``permutation``. For example::

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PermutationGate
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig, HighLevelSynthesis
        from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPluginManager

        # Create a permutation and add it to a quantum circuit
        perm = PermutationGate([4, 6, 3, 7, 1, 2, 0, 5])
        qc = QuantumCircuit(8)
        qc.append(perm, range(8))

        # KMSSynthesisPermutation plugin for permutations
        # Returns a quantum circuit with size 18 and depth 6
        # but adhering to the linear nearest-neighbor architecture.
        qct = PassManager(HighLevelSynthesis(HLSConfig(permutation=[("kms", {})]))).run(qc)
        print(f"kms: {qct.size() = }, {qct.depth() = }")
    """

    def run(self, high_level_object, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin can be accessed by the ``basic`` method name in the
    ``HLSConfig`` for ``permutation``. For example::

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PermutationGate
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig, HighLevelSynthesis
        from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPluginManager

        # Create a permutation and add it to a quantum circuit
        perm = PermutationGate([4, 6, 3, 7, 1, 2, 0, 5])
        qc = QuantumCircuit(8)
        qc.append(perm, range(8))

        # BasicSynthesisPermutation plugin for permutations
        # Returns a quantum circuit with size 6 and depth 3
        qct = PassManager(HighLevelSynthesis(HLSConfig(permutation=[("basic", {})]))).run(qc)
        print(f"basic: {qct.size() = }, {qct.depth() = }")
    """

    def run(self, high_level_object, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin can be accessed by the ``acg`` method name in the
    ``HLSConfig`` for ``permutation``. For example::

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PermutationGate
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig, HighLevelSynthesis
        from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPluginManager

        # Create a permutation and add it to a quantum circuit
        perm = PermutationGate([4, 6, 3, 7, 1, 2, 0, 5])
        qc = QuantumCircuit(8)
        qc.append(perm, range(8))

        # ACGSynthesisPermutation plugin for permutations
        # Returns a quantum circuit with size 6 and depth 2
        qct = PassManager(HighLevelSynthesis(HLSConfig(permutation=[("acg", {})]))).run(qc)
        print(f"acg: {qct.size() = }, {qct.depth() = }")
    """

    def run(self, high_level_object, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition
