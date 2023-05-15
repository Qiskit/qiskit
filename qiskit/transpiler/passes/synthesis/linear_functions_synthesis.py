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


"""Synthesize LinearFunctions."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HighLevelSynthesis, HLSConfig
from qiskit.utils.deprecation import deprecate_func


class LinearFunctionsSynthesis(HighLevelSynthesis):
    """DEPRECATED: Synthesize linear functions.

    Under the hood, this runs the default high-level synthesis plugin for linear functions.
    """

    @deprecate_func(
        additional_msg="Instead, use :class:`~.HighLevelSynthesis`.",
        since="0.23.0",
    )
    def __init__(self):
        # This config synthesizes only linear functions using the "default" method.
        default_linear_config = HLSConfig(
            linear_function=[("default", {})],
            use_default_on_unspecified=False,
        )
        super().__init__(hls_config=default_linear_config)


class LinearFunctionsToPermutations(TransformationPass):
    """Promotes linear functions to permutations when possible."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LinearFunctionsToPermutations pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with LinearFunctions synthesized.
        """

        for node in dag.named_nodes("linear_function"):
            try:
                pattern = node.op.permutation_pattern()
            except CircuitError:
                continue

            permutation = PermutationGate(pattern)
            dag.substitute_node(node, permutation)
        return dag
