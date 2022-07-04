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
An AQC synthesis plugin to Qiskit's transpiler.
"""
import numpy as np

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin


class AQCSynthesisPlugin(UnitarySynthesisPlugin):
    """
    An AQC-based Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"aqc"``.

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
        """Maximum number of supported qubits is ``14``."""
        return 14

    @property
    def min_qubits(self):
        """Minimum number of supported qubits is ``3``."""
        return 3

    @property
    def supports_natural_direction(self):
        """The plugin does not support natural direction,
        it assumes bidirectional two qubit gates."""
        return False

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
        """The plugin does not support basis gates and by default it synthesizes a circuit using
        ``["rx", "ry", "rz", "cx"]`` gate basis."""
        return False

    @property
    def supports_coupling_map(self):
        """The plugin does not support coupling maps."""
        return False

    def run(self, unitary, **options):

        # Runtime imports to avoid the overhead of these imports for
        # plugin discovery and only use them if the plugin is run/used
        from qiskit.algorithms.optimizers import L_BFGS_B
        from qiskit.transpiler.synthesis.aqc.aqc import AQC
        from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
        from qiskit.transpiler.synthesis.aqc.cnot_unit_circuit import CNOTUnitCircuit
        from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective

        num_qubits = int(round(np.log2(unitary.shape[0])))

        config = options.get("config") or {}

        network_layout = config.get("network_layout", "spin")
        connectivity_type = config.get("connectivity_type", "full")
        depth = config.get("depth", 0)

        cnots = make_cnot_network(
            num_qubits=num_qubits,
            network_layout=network_layout,
            connectivity_type=connectivity_type,
            depth=depth,
        )

        optimizer = config.get("optimizer", L_BFGS_B(maxiter=1000))
        seed = config.get("seed")
        aqc = AQC(optimizer, seed)

        approximate_circuit = CNOTUnitCircuit(num_qubits=num_qubits, cnots=cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)

        initial_point = config.get("initial_point")
        aqc.compile_unitary(
            target_matrix=unitary,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
            initial_point=initial_point,
        )

        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit
