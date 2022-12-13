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
"""
A Solovay-Kitaev synthesis plugin to Qiskit's transpiler.
"""

from __future__ import annotations

import numpy as np

from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
    generate_basic_approximations,
)
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from .plugin import UnitarySynthesisPlugin


class SolovayKitaev(TransformationPass):
    r"""Approximately decompose 1q gates to a discrete basis using the Solovay-Kitaev algorithm.

    See :mod:`qiskit.synthesis.discrete_basis` for more information.
    """

    def __init__(
        self,
        recursion_degree: int = 3,
        basic_approximations: str | dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Args:
            recursion_degree: The recursion depth for the Solovay-Kitaev algorithm.
                A larger recursion depth increases the accuracy and length of the
                decomposition.
            basic_approximations: The basic approximations for the finding the best discrete
                decomposition at the root of the recursion. If a string, it specifies the ``.npy``
                file to load the approximations from. If a dictionary, it contains
                ``{label: SO(3)-matrix}`` pairs. If None, a default based on the H, T and Tdg gates
                up to combinations of depth 10 is generated.
        """
        super().__init__()
        self.recursion_degree = recursion_degree
        self._sk = SolovayKitaevDecomposition(basic_approximations)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ``SolovayKitaev`` pass on `dag`.

        Args:
            dag: The input dag.

        Returns:
            Output dag with 1q gates synthesized in the discrete target basis.

        Raises:
            TranspilerError: if a gates does not have to_matrix
        """
        for node in dag.op_nodes():
            if not node.op.num_qubits == 1:
                continue  # ignore all non-single qubit gates

            if not hasattr(node.op, "to_matrix"):
                raise TranspilerError(
                    "SolovayKitaev does not support gate without "
                    f"to_matrix method: {node.op.name}"
                )

            matrix = node.op.to_matrix()

            # call solovay kitaev
            approximation = self._sk.run(matrix, self.recursion_degree, return_dag=True)

            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, approximation)

        return dag


class SolovayKitaevSynthesis(UnitarySynthesisPlugin):
    """A Solovay-Kitaev Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"sk"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    basis_approximations (str | dict):
        The basic approximations for the finding the best discrete decomposition at the root of the
        recursion. If a string, it specifies the ``.npy`` file to load the approximations from.
        If a dictionary, it contains ``{label: SO(3)-matrix}`` pairs. If None, a default based on
        the specified ``basis_gates`` and ``depth`` is generated.

    basis_gates (list):
        A list of strings specifying the discrete basis gates to decompose to. If None,
        defaults to ``["h", "t", "tdg"]``.

    depth (int):
        The gate-depth of the the basic approximations. All possible, unique combinations of the
        basis gates up to length ``depth`` are considered. If None, defaults to 10.

    recursion_degree (int):
        The number of times the decomposition is recursively improved. If None, defaults to 3.
    """

    # we cache an instance of the Solovay-Kitaev class to generate the
    # computationally expensive basis approximation of single qubit gates only once
    _sk = None

    @property
    def max_qubits(self):
        """Maximum number of supported qubits is ``1``."""
        return 1

    @property
    def min_qubits(self):
        """Minimum number of supported qubits is ``1``."""
        return 1

    @property
    def supports_natural_direction(self):
        """The plugin does not support natural direction, it does not assume
        bidirectional two qubit gates."""
        return True

    @property
    def supports_pulse_optimize(self):
        """The plugin does not support optimization of pulses."""
        return False

    @property
    def supports_gate_lengths(self):
        """The plugin does not support gate lengths."""
        return False

    @property
    def supports_gate_errors(self):
        """The plugin does not support gate errors."""
        return False

    @property
    def supported_bases(self):
        """The plugin does not support bases for synthesis."""
        return None

    @property
    def supports_basis_gates(self):
        """The plugin does not support basis gates. By default it synthesis to the
        ``["h", "t", "tdg"]`` gate basis."""
        return True

    @property
    def supports_coupling_map(self):
        """The plugin does not support coupling maps."""
        return False

    def run(self, unitary, **options):

        # Runtime imports to avoid the overhead of these imports for
        # plugin discovery and only use them if the plugin is run/used
        config = options.get("config") or {}

        recursion_degree = config.get("recursion_degree", 3)

        # if we didn't yet construct the Solovay-Kitaev instance, which contains
        # the basic approximations, do it now
        if SolovayKitaevSynthesis._sk is None:
            basic_approximations = config.get("basic_approximations", None)
            basis_gates = options.get("basis_gates", ["h", "t", "tdg"])

            # if the basic approximations are not generated and not given,
            # try to generate them if the basis set is specified
            if basic_approximations is None:
                depth = config.get("depth", 10)
                basic_approximations = generate_basic_approximations(basis_gates, depth)

            SolovayKitaevSynthesis._sk = SolovayKitaevDecomposition(basic_approximations)

        approximate_circuit = SolovayKitaevSynthesis._sk.run(unitary, recursion_degree)
        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit
