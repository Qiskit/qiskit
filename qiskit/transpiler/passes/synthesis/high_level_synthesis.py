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


from typing import Union

from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.synthesis import synth_permutation_basic, synth_permutation_acg
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.synthesis.clifford import synth_clifford_full
from qiskit.synthesis.linear import synth_cnot_count_full_pmh
from qiskit.synthesis.permutation import synth_permutation_depth_lnn_kms
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, ControlModifier, PowerModifier

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
    """Synthesize higher-level objects.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`)
    and annotated operations (objects of type :class:`.AnnotatedOperation`).
    By default, all higher-level objects are synthesized, so the output is a DAG without such objects.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

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

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).
    """

    def __init__(self, hls_config=None):
        super().__init__()

        if hls_config is not None:
            self.hls_config = hls_config
        else:
            # When the config file is not provided, we will use the "default" method
            # to synthesize Operations (when available).
            self.hls_config = HLSConfig(True)

        self.hls_plugin_manager = HighLevelSynthesisPluginManager()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations synthesized.

        Raises:
            TranspilerError: when the transpiler is unable to synthesize the given DAG
            (for instance, when the specified synthesis method is not available).
        """

        # The pass is recursive, as we may have annotated gates whose definitions
        # consist of other annotated gates, whose definitions include for instance
        # LinearFunctions. Note that in order to synthesize a controlled linear
        # function, we must first fully synthesize the linear function, and then
        # synthesize the circuit obtained by adding control logic.

        # copy dag_op_nodes because we are modifying the DAG below
        dag_op_nodes = dag.op_nodes()

        for node in dag_op_nodes:
            decomposition = self._recursively_handle_op(node.op)

            if not isinstance(decomposition, (QuantumCircuit, Operation)):
                raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {node.op}.")

            if isinstance(decomposition, QuantumCircuit):
                dag.substitute_node_with_dag(node, circuit_to_dag(decomposition))
            elif isinstance(decomposition, Operation):
                dag.substitute_node(node, decomposition)

        return dag

    def _recursively_handle_op(self, op: Operation) -> Union[Operation, QuantumCircuit]:
        """Recursively synthesizes a single operation.

        The result can be either another operation or a quantum circuit.

        Some examples when the result can be another operation:
        Adding control to CX-gate results in CCX-gate,
        Adding inverse to SGate results in SdgGate.

        Some examples when the result can be a quantum circuit:
        Synthesizing a LinearFunction produces a quantum circuit consisting of
        CX-gates.

        The function recursively handles operation's definition, if it exists.
        """

        # First, try to apply plugin mechanism
        decomposition = self._synthesize_op_using_plugins(op)
        if decomposition:
            return decomposition

        # Second, handle annotated operations
        decomposition = self._synthesize_annotated_op(op)
        if decomposition:
            return decomposition

        # Third, recursively descend into op's definition if exists
        if getattr(op, "definition", None) is not None:
            dag = circuit_to_dag(op.definition)
            dag = self.run(dag)
            op.definition = dag_to_circuit(dag)

        return op

    def _synthesize_op_using_plugins(self, op: Operation) -> Union[QuantumCircuit, None]:
        """
        Attempts to synthesize op using plugin mechanism.
        Returns either the synthesized circuit or None (which occurs when no
        synthesis methods are available or specified).
        """
        hls_plugin_manager = self.hls_plugin_manager

        if op.name in self.hls_config.methods.keys():
            # the operation's name appears in the user-provided config,
            # we use the list of methods provided by the user
            methods = self.hls_config.methods[op.name]
        elif (
            self.hls_config.use_default_on_unspecified
            and "default" in hls_plugin_manager.method_names(op.name)
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
                if plugin_specifier not in hls_plugin_manager.method_names(op.name):
                    raise TranspilerError(
                        "Specified method: %s not found in available plugins for %s"
                        % (plugin_specifier, op.name)
                    )
                plugin_method = hls_plugin_manager.method(op.name, plugin_specifier)
            else:
                plugin_method = plugin_specifier

            # ToDo: similarly to UnitarySynthesis, we should pass additional parameters
            #       e.g. coupling_map to the synthesis algorithm.
            # print(f"{plugin_method = }, {op = }, {plugin_args = }")
            decomposition = plugin_method.run(op, **plugin_args)

            # The synthesis methods that are not suited for the given higher-level-object
            # will return None, in which case the next method in the list will be used.
            if decomposition is not None:
                return decomposition

        return None

    def _synthesize_annotated_op(self, op: Operation) -> Union[Operation, None]:
        """
        Recursively synthesizes annotated operations.
        Returns either the synthesized operation or None (which occurs when the operation
        is not an annotated operation).
        """
        if isinstance(op, AnnotatedOperation):
            synthesized_op = self._recursively_handle_op(op.base_op)

            if not synthesized_op:
                raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {op.base_op}.")
            if not isinstance(synthesized_op, (QuantumCircuit, Gate)):
                raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {op.base_op}.")

            for modifier in op.modifiers:
                if isinstance(modifier, InverseModifier):
                    # ToDo: what do we do for clifford or Operation without inverse method?
                    synthesized_op = synthesized_op.inverse()
                elif isinstance(modifier, ControlModifier):
                    # Above we checked that we either have a gate or a quantum circuit
                    synthesized_op = synthesized_op.control(
                        num_ctrl_qubits=modifier.num_ctrl_qubits,
                        label=None,
                        ctrl_state=modifier.ctrl_state,
                    )
                elif isinstance(modifier, PowerModifier):
                    if isinstance(synthesized_op, QuantumCircuit):
                        qc = synthesized_op
                    else:
                        qc = QuantumCircuit(synthesized_op.num_qubits, synthesized_op.num_clbits)
                        qc.append(synthesized_op, range(synthesized_op.num_qubits), range(synthesized_op.num_clbits))

                    qc = qc.power(modifier.power)
                    synthesized_op = qc.to_gate()
                else:
                    raise TranspilerError(f"Unknown modifier {modifier}.")

            return synthesized_op
        return None


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
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method."""

    def run(self, high_level_object, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition
