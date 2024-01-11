# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fake OpenQASM simulator.
"""

from qiskit.providers.models import GateConfig, QasmBackendConfiguration

from .fake_backend import FakeBackend


class FakeQasmSimulator(FakeBackend):
    """A fake simulator backend."""

    def __init__(self):
        configuration = QasmBackendConfiguration(
            backend_name="fake_qasm_simulator",
            backend_version="0.0.0",
            n_qubits=5,
            basis_gates=["u1", "u2", "u3", "cx", "id", "unitary"],
            coupling_map=None,
            simulator=True,
            local=True,
            conditional=True,
            open_pulse=False,
            memory=True,
            max_shots=65536,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            dt=1.3333,
            dtm=10.5,
            qubit_lo_range=[[4.95, 5.05], [4.95, 5.05], [4.95, 5.05], [4.95, 5.05], [4.95, 5.05]],
            meas_lo_range=[[6.65, 6.75], [6.65, 6.75], [6.65, 6.75], [6.65, 6.75], [6.65, 6.75]],
        )

        super().__init__(configuration)

    def properties(self):
        return None
