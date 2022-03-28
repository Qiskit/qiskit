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

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

from .solovay_kitaev import SolovayKitaev
from .generate_basis_approximations import generate_basic_approximations

# we globally cache an instance of the Solovay-Kitaev class to generate the
# computationally expensive basis approximation of single qubit gates only once
SK_ = None


class SolovayKitaevSynthesisPlugin(UnitarySynthesisPlugin):
    """
    An Solovay-Kitaev-based Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"sk"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    network_layout (str)
        Type of network geometry, one of {``"sequ"``, ``"spin"``, ``"cart"``, ``"cyclic_spin"``,
        ``"cyclic_line"``}. Default value is ``"spin"``.

    connectivity_type (str)
        type of inter-qubit connectivity, {``"full"``, ``"line"``, ``"star"``}.  Default value
        is ``"full"``.

    depth (int)
        depth of the CNOT-network, i.e. the number of layers, where each layer consists of a
        single CNOT-block.

    optimizer (:class:`~qiskit.algorithms.optimizers.Optimizer`)
        An instance of optimizer to be used in the optimization process.

    seed (int)
        A random seed.

    initial_point (:class:`~numpy.ndarray`)
        Initial values of angles/parameters to start the optimization process from.
    """

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
        global SK_  # pylint: disable=global-statement

        if SK_ is None:
            basic_approximations = config.get("basic_approximations", None)
            basis_gates = options.get("basis_gates", None)

            # if the basic approximations are not generated and not given,
            # try to generate them if the basis set is specified
            if basic_approximations is None and basis_gates is not None:
                depth = config.get("depth", 10)
                basic_approximations = generate_basic_approximations(basis_gates, depth)

                SK_ = SolovayKitaev(basic_approximations)

        approximate_circuit = SK_.run(unitary, recursion_degree)
        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit
