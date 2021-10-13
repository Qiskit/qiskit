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

from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
from qiskit.transpiler.synthesis.aqc.cnot_unit_circuit import CNOTUnitCircuit
from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective


class AQCSynthesisPlugin(UnitarySynthesisPlugin):
    """An AQC-based Qiskit unitary synthesis plugin."""

    @property
    def max_qubits(self):
        """Maximum number of supported qubits is ``10``."""
        return 10

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
        num_qubits = int(round(np.log2(unitary.shape[0])))

        layout = options.get("layout") or "spin"
        connectivity = options.get("connectivity") or "full"
        depth = int(options.get("depth") or 0)
        cnots = make_cnot_network(
            num_qubits=num_qubits,
            network_layout=layout,
            connectivity_type=connectivity,
            depth=depth,
        )

        seed = options.get("seed")
        max_iter = options.get("max_iter") or 1000
        optimizer = L_BFGS_B(maxiter=max_iter)
        aqc = AQC(optimizer, seed)

        name = options.get("approx_name") or "aqc"
        approximate_circuit = CNOTUnitCircuit(num_qubits=num_qubits, cnots=cnots, name=name)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)

        aqc.compile_unitary(
            target_matrix=unitary,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
        )

        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit
