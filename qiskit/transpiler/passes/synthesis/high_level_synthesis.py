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

from typing import Union, List

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.coupling import CouplingMap
from qiskit.synthesis.clifford import synth_clifford_full
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.linear.linear_circuits_utils import optimize_cx_4_options, _compare_circuits
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

    def __init__(self, coupling_map: CouplingMap = None, hls_config=None):
        """
        HighLevelSynthesis initializer.

        Args:
            coupling_map (CouplingMap): the coupling map of the backend
                in case synthesis is done on a physical circuit.
            hls_config (HLSConfig): the high-level-synthesis config file
            specifying synthesis methods and parameters.
        """
        super().__init__()

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
        hls_plugin_manager = HighLevelSynthesisPluginManager()

        dag_bit_indices = {bit: i for i, bit in enumerate(dag.qubits)}

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

                # There are two ways to specify an individual method being run, either a tuple
                #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}),
                # or as a class instance
                #   KMSSynthesisLinearFunction(all_mats=1, max_paths=100, orig_circuit=0).
                if isinstance(method, tuple):
                    plugin_name, plugin_args = method

                    if plugin_name not in hls_plugin_manager.method_names(node.name):
                        raise TranspilerError(
                            "Specified method: %s not found in available plugins for %s"
                            % (plugin_name, node.name)
                        )

                    plugin_method = hls_plugin_manager.method(node.name, plugin_name)

                else:
                    plugin_method = method
                    plugin_args = {}

                if self._coupling_map:
                    plugin_args["coupling_map"] = self._coupling_map
                    plugin_args["qubits"] = [dag_bit_indices[x] for x in node.qargs]

                decomposition = plugin_method.run(node.op, **plugin_args)

                # The above synthesis method may return:
                # - None, if the synthesis algorithm is not suited for the given higher-level-object
                #   (in which case we consider the next method in the list if available).
                # - decomposition when the order of qubits is not important
                # - a tuple (decomposition, wires) when the node's qubits need to be reordered
                if decomposition is not None:
                    if isinstance(decomposition, tuple):
                        decomposition_dag = circuit_to_dag(decomposition[0])
                        wires = [decomposition_dag.wires[i] for i in decomposition[1]]
                        dag.substitute_node_with_dag(node, decomposition_dag, wires=wires)
                    else:
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
        # For now, use PMH algorithm by default
        decomposition = PMHSynthesisLinearFunction().run(high_level_object, **options)
        return decomposition


class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method."""

    def __init__(self, **options):
        self._options = options

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""

        # Combine the options passed in the initializer and now,
        # prioritizing values passed now.
        run_options = self._options.copy()
        run_options.update(options)

        # options supported by this plugin
        coupling_map = run_options.get("coupling_map", None)
        qubits = run_options.get("qubits", None)
        consider_all_mats = run_options.get("all_mats", 0)
        max_paths = run_options.get("max_paths", 1)
        consider_original_circuit = run_options.get("orig_circuit", 1)
        optimize_count = run_options.get("opt_count", 1)

        # At the end, if not none, represents the best decomposition adhering to LNN architecture.
        best_decomposition = None

        # At the end, if not none, represents the path of qubits through the coupling map
        # over which the LNN synthesis is applied.
        best_path = None

        if consider_original_circuit:
            best_decomposition = high_level_object.original_circuit

        if not coupling_map:
            if not consider_all_mats:
                decomposition = synth_cnot_depth_line_kms(high_level_object.linear)
            else:
                decomposition = optimize_cx_4_options(
                    synth_cnot_depth_line_kms,
                    high_level_object.linear,
                    optimize_count=optimize_count,
                )

            if not best_decomposition or _compare_circuits(
                best_decomposition, decomposition, optimize_count=optimize_count
            ):
                best_decomposition = decomposition

        else:
            # Consider the coupling map over the qubits on which the linear function is applied.
            reduced_map = coupling_map.reduce(qubits)

            # Find one or more paths through the coupling map (when such exist).
            considered_paths = _hamiltonian_paths(reduced_map, max_paths)

            for path in considered_paths:
                permuted_linear_function = high_level_object.permute(path)

                if not consider_all_mats:
                    decomposition = synth_cnot_depth_line_kms(permuted_linear_function.linear)
                else:
                    decomposition = optimize_cx_4_options(
                        synth_cnot_depth_line_kms,
                        permuted_linear_function.linear,
                        optimize_count=False,
                    )

                if not best_decomposition or _compare_circuits(
                    best_decomposition, decomposition, optimize_count=False
                ):
                    best_decomposition = decomposition
                    best_path = path

        if best_path is None:
            return best_decomposition

        return best_decomposition, best_path


class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method."""

    def __init__(self, **options):
        self._options = options

    def run(self, high_level_object, **options):
        """Run synthesis for the given LinearFunction."""

        # Combine the options passed in the initializer and now,
        # prioritizing values passed now.
        # Note: run_options = self._options | options is only supported for python >= 3.9.
        run_options = self._options.copy()
        run_options.update(options)

        # options supported by this plugin
        coupling_map = run_options.get("coupling_map", None)
        consider_all_mats = run_options.get("all_mats", 0)
        consider_original_circuit = run_options.get("orig_circuit", 1)
        optimize_count = run_options.get("opt_count", 1)

        # At the end, if not none, represents the best decomposition.
        best_decomposition = None

        if consider_original_circuit:
            best_decomposition = high_level_object.original_circuit

        # This synthesis method is not aware of the coupling map, so we cannot apply
        # this method when the coupling map is not None.
        # (Though, technically, we could check if the reduced coupling map is
        # fully-connected).

        if not coupling_map:
            if not consider_all_mats:
                decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
            else:
                decomposition = optimize_cx_4_options(
                    synth_cnot_count_full_pmh,
                    high_level_object.linear,
                    optimize_count=optimize_count,
                )

            if not best_decomposition or _compare_circuits(
                best_decomposition, decomposition, optimize_count=optimize_count
            ):
                best_decomposition = decomposition

        return best_decomposition


def _hamiltonian_paths(
    coupling_map: CouplingMap, cutoff: Union[None, int] = None
) -> List[List[int]]:
    """Returns a list of all Hamiltonian paths in ``coupling_map`` (stopping the enumeration when
    the number of already discovered paths exceeds the ``cutoff`` value, when specified).
    In particular, returns an empty list if there are no Hamiltonian paths.
    """

    # This is a temporary function, the plan is to move it to rustworkx

    def _should_stop():
        return cutoff is not None and len(all_paths) >= cutoff

    def _recurse(current_node):
        current_path.append(current_node)
        on_path[current_node] = True

        if len(current_path) == coupling_map.size():
            # Discovered a new Hamiltonian path
            all_paths.append(current_path.copy())

        if _should_stop():
            return

        unvisited_neighbors = [
            node for node in coupling_map.neighbors(current_node) if not on_path[node]
        ]
        for node in unvisited_neighbors:
            _recurse(node)
            if _should_stop():
                return

        current_path.pop()
        on_path[current_node] = False

    all_paths = []
    qubits = coupling_map.physical_qubits
    current_path = []
    on_path = [False] * len(qubits)

    for qubit in qubits:
        _recurse(qubit)
        if _should_stop():
            break
    return all_paths
