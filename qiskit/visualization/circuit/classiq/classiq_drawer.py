# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classiq circuit drawer"""


import webbrowser
from threading import Lock
from typing import Optional

from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
from qiskit.visualization.circuit.classiq.authentication.token_manager import TokenManager
from qiskit.visualization.circuit.classiq.draw_exceptions import (
    ClassiqQASMException,
    ClassiqCircuitIDNotFoundException,
)

from qiskit.utils import optionals as _optionals


class ClassiqCircuitDrawer:
    """Create a new Classiq drawer to open a quantum circuit with the Classiq visualizer.

    Args:
        circuit (QuantumCircuit): a quantum circuit
    """

    _CLASSIQ_PLATFORM_URL = "https://platform.classiq.io"
    _QASM_PATH = "/api/v1/analyzer/tasks/qasm"
    _token_manager: Optional[TokenManager] = None
    _token_manager_lock = Lock()

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._access_token: Optional[str] = None
        with self._token_manager_lock:
            if not ClassiqCircuitDrawer._token_manager:
                ClassiqCircuitDrawer._token_manager = TokenManager()

    @property
    def qasm(self):
        """Return the circuit QASM string"""
        return qasm2.dumps(self._circuit)

    def _authenticate_to_classiq(self):
        """Authenticate to Classiq platform"""
        if self._token_manager.get_access_token() is None:
            self._token_manager.authenticate()
        self._access_token = self._token_manager.get_access_token()

    @_optionals.HAS_REQUESTS.require_in_call
    def _send_qasm_to_classiq(self):
        """Send the QASM code to the Classiq platform

        Returns:
            str: The circuit id in Classiq of the uploaded qasm.
        """

        import requests

        payload = {"qasm": self.qasm}
        response = requests.post(
            url=self._CLASSIQ_PLATFORM_URL + self._QASM_PATH,
            headers=self.get_headers(),
            json=payload,
            timeout=30,
        )
        data = response.json()
        code = response.status_code
        if code == 200:
            return data["id"]
        elif not response.ok:
            raise ClassiqQASMException(code, data["detail"])
        raise ClassiqCircuitIDNotFoundException(code)

    def _open_classiq_ide_with_circuit(self, circuit_id):
        """Open the Classiq platform's IDE with the corresponding circuit

        Args:
            circuit_id (str): The circuit identifier in Classiq to visualize
        """
        webbrowser.open(f"{self._CLASSIQ_PLATFORM_URL}/circuit/{circuit_id}")

    def draw(self):
        """Draw the circuit in Classiq's analyzer"""
        self._authenticate_to_classiq()
        circuit_id = self._send_qasm_to_classiq()
        self._open_classiq_ide_with_circuit(circuit_id)

    def get_headers(self):
        """Get headers for sending payload"""
        return {"Authorization": f"Bearer {self._access_token}"}
