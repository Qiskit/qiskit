# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Contains a (slow) python statevector simulator.

It simulates the statevector through a quantum circuit. It is exponential in
the number of qubits.

We advise using the c++ simulator or online simulator for larger size systems.

The input is a qobj dictionary and the output is a Result object.

The input qobj to this simulator has no shots, no measures, no reset, no noise.
"""

import logging
from qiskit.providers.basicaer.exceptions import BasicAerError
from qiskit.providers.models import QasmBackendConfiguration
from .qasm_simulator import QasmSimulatorPy

logger = logging.getLogger(__name__)


class StatevectorSimulatorPy(QasmSimulatorPy):
    """Python statevector simulator."""

    DEFAULT_CONFIGURATION = {
        "backend_name": "statevector_simulator",
        "backend_version": "1.1.0",
        "n_qubits": 24,
        "url": "https://github.com/Qiskit/qiskit-terra",
        "simulator": True,
        "local": True,
        "conditional": True,
        "open_pulse": False,
        "memory": True,
        "max_shots": 0,
        "coupling_map": None,
        "description": "A Python statevector simulator for qobj files",
        "basis_gates": ["u1", "u2", "u3", "rz", "sx", "x", "cx", "id", "unitary"],
        "gates": [
            {
                "name": "u1",
                "parameters": ["lambda"],
                "qasm_def": "gate u1(lambda) q { U(0,0,lambda) q; }",
            },
            {
                "name": "u2",
                "parameters": ["phi", "lambda"],
                "qasm_def": "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }",
            },
            {
                "name": "u3",
                "parameters": ["theta", "phi", "lambda"],
                "qasm_def": "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }",
            },
            {"name": "rz", "parameters": ["phi"], "qasm_def": "gate rz(phi) q { U(0,0,phi) q; }"},
            {
                "name": "sx",
                "parameters": [],
                "qasm_def": "gate sx(phi) q { U(pi/2,7*pi/2,pi/2) q; }",
            },
            {"name": "x", "parameters": [], "qasm_def": "gate x q { U(pi,7*pi/2,pi/2) q; }"},
            {"name": "cx", "parameters": [], "qasm_def": "gate cx c,t { CX c,t; }"},
            {"name": "id", "parameters": [], "qasm_def": "gate id a { U(0,0,0) a; }"},
            {"name": "unitary", "parameters": ["matrix"], "qasm_def": "unitary(matrix) q1, q2,..."},
        ],
    }

    # Override base class value to return the final state vector
    SHOW_FINAL_STATE = True

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(
            configuration=(
                configuration or QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION)
            ),
            provider=provider,
            **fields,
        )

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. No shots
        2. No measurements in the middle
        """
        num_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if num_qubits > max_qubits:
            raise BasicAerError(
                f"Number of qubits {num_qubits} is greater than maximum ({max_qubits}) "
                f'for "{self.name()}".'
            )
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.', self.name())
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            name = experiment.header.name
            if getattr(experiment.config, "shots", 1) != 1:
                logger.info(
                    '"%s" only supports 1 shot. Setting shots=1 for circuit "%s".',
                    self.name(),
                    name,
                )
                experiment.config.shots = 1
