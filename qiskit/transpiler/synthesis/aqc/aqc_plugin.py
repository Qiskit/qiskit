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

    def __init__(self) -> None:
        super().__init__()

        # define defaults
        self._layout = "spin"
        self._layout = "spin"
        self._connectivity = "full"
        self._depth = 0
        self._seed = 12345
        self._maxiter = 1000

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
        num_qubits = int(round(np.log2(unitary.shape[0])))

        cnots = make_cnot_network(
            num_qubits=num_qubits,
            network_layout=self._layout,
            connectivity_type=self._connectivity,
            depth=self._depth,
        )

        optimizer = L_BFGS_B(maxiter=self._maxiter)
        aqc = AQC(optimizer, self._seed)

        approximate_circuit = CNOTUnitCircuit(num_qubits=num_qubits, cnots=cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)

        aqc.compile_unitary(
            target_matrix=unitary,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
        )

        dag_circuit = circuit_to_dag(approximate_circuit)
        return dag_circuit
