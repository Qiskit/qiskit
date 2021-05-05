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

# pylint: disable=no-name-in-module,import-error

"""
Base class for dummy backends.
"""

import uuid
import warnings

from qiskit import circuit
from qiskit.providers.models import BackendProperties
from qiskit.providers import BackendV1, BaseBackend
from qiskit import pulse
from qiskit.exceptions import QiskitError
from qiskit.test.mock import fake_job

try:
    from qiskit.providers import aer

    HAS_AER = True
except ImportError:
    HAS_AER = False
    from qiskit.providers import basicaer


class _Credentials:
    def __init__(self, token="123456", url="https://"):
        self.token = token
        self.url = url
        self.hub = "hub"
        self.group = "group"
        self.project = "project"


class FakeBackend(BackendV1):
    """This is a dummy backend just for testing purposes."""

    def __init__(self, configuration, time_alive=10):
        """FakeBackend initializer.

        Args:
            configuration (BackendConfiguration): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration)
        self.time_alive = time_alive
        self._credentials = _Credentials()

    def properties(self):
        """Return backend properties"""
        coupling_map = self.configuration().coupling_map
        unique_qubits = list(set().union(*coupling_map))

        properties = {
            "backend_name": self.name(),
            "backend_version": self.configuration().backend_version,
            "last_update_date": "2000-01-01 00:00:00Z",
            "qubits": [
                [
                    {"date": "2000-01-01 00:00:00Z", "name": "T1", "unit": "\u00b5s", "value": 0.0},
                    {"date": "2000-01-01 00:00:00Z", "name": "T2", "unit": "\u00b5s", "value": 0.0},
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 0.0,
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.0,
                    },
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 1},
                ]
                for _ in range(len(unique_qubits))
            ],
            "gates": [
                {
                    "gate": "cx",
                    "name": "CX" + str(pair[0]) + "_" + str(pair[1]),
                    "parameters": [
                        {
                            "date": "2000-01-01 00:00:00Z",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [pair[0], pair[1]],
                }
                for pair in coupling_map
            ],
            "general": [],
        }

        return BackendProperties.from_dict(properties)

    @classmethod
    def _default_options(cls):
        if HAS_AER:
            return aer.QasmSimulator._default_options()
        else:
            return basicaer.QasmSimulatorPy._default_options()

    def run(self, run_input, **kwargs):
        """Main job in simulator"""
        circuits = run_input
        pulse_job = None
        if isinstance(circuits, pulse.Schedule):
            pulse_job = True
        elif isinstance(circuits, circuit.QuantumCircuit):
            pulse_job = False
        elif isinstance(circuits, list):
            if circuits:
                if all(isinstance(x, pulse.Schedule) for x in circuits):
                    pulse_job = True
                elif all(isinstance(x, circuit.QuantumCircuit) for x in circuits):
                    pulse_job = False
        if pulse_job is None:
            raise QiskitError(
                "Invalid input object %s, must be either a "
                "QuantumCircuit, Schedule, or a list of "
                "either" % circuits
            )
        if HAS_AER:
            if pulse_job:
                from qiskit.providers.aer.pulse import PulseSystemModel

                system_model = PulseSystemModel.from_backend(self)
                sim = aer.Aer.get_backend("pulse_simulator")
                job = sim.run(circuits, system_model, **kwargs)
            else:
                sim = aer.Aer.get_backend("qasm_simulator")
                if self.properties():
                    from qiskit.providers.aer.noise import NoiseModel

                    noise_model = NoiseModel.from_backend(self, warnings=False)
                    job = sim.run(circuits, noise_model=noise_model)
                else:
                    job = sim.run(circuits, **kwargs)
        else:
            if pulse_job:
                raise QiskitError("Unable to run pulse schedules without " "qiskit-aer installed")
            warnings.warn("Aer not found using BasicAer and no noise", RuntimeWarning)
            sim = basicaer.BasicAer.get_backend("qasm_simulator")
            job = sim.run(circuits, **kwargs)
        return job


class FakeLegacyBackend(BaseBackend):
    """This is a dummy backend just for testing purposes of the legacy providers interface."""

    def __init__(self, configuration, time_alive=10):
        """FakeBackend initializer.
        Args:
            configuration (BackendConfiguration): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration)
        self.time_alive = time_alive
        self._credentials = _Credentials()

    def properties(self):
        """Return backend properties"""
        coupling_map = self.configuration().coupling_map
        unique_qubits = list(set().union(*coupling_map))

        properties = {
            "backend_name": self.name(),
            "backend_version": self.configuration().backend_version,
            "last_update_date": "2000-01-01 00:00:00Z",
            "qubits": [
                [
                    {"date": "2000-01-01 00:00:00Z", "name": "T1", "unit": "\u00b5s", "value": 0.0},
                    {"date": "2000-01-01 00:00:00Z", "name": "T2", "unit": "\u00b5s", "value": 0.0},
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 0.0,
                    },
                    {
                        "date": "2000-01-01 00:00:00Z",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.0,
                    },
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 1},
                ]
                for _ in range(len(unique_qubits))
            ],
            "gates": [
                {
                    "gate": "cx",
                    "name": "CX" + str(pair[0]) + "_" + str(pair[1]),
                    "parameters": [
                        {
                            "date": "2000-01-01 00:00:00Z",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [pair[0], pair[1]],
                }
                for pair in coupling_map
            ],
            "general": [],
        }

        return BackendProperties.from_dict(properties)

    def run(self, qobj):
        """Main job in simulator"""
        if HAS_AER:
            if qobj.type == "PULSE":
                from qiskit.providers.aer.pulse import PulseSystemModel

                system_model = PulseSystemModel.from_backend(self)
                sim = aer.Aer.get_backend("pulse_simulator")
                job = sim.run(qobj, system_model)
            else:
                sim = aer.Aer.get_backend("qasm_simulator")
                if self.properties():
                    from qiskit.providers.aer.noise import NoiseModel

                    noise_model = NoiseModel.from_backend(self, warnings=False)
                    job = sim.run(qobj, noise_model=noise_model)
                else:
                    job = sim.run(qobj)

            out_job = fake_job.FakeLegacyJob(self, job.job_id, None)
            out_job._future = job._future
        else:
            if qobj.type == "PULSE":
                raise QiskitError("Unable to run pulse schedules without " "qiskit-aer installed")
            warnings.warn("Aer not found using BasicAer and no noise", RuntimeWarning)

            def run_job():
                sim = basicaer.BasicAer.get_backend("qasm_simulator")
                return sim.run(qobj).result()

            job_id = uuid.uuid4()
            out_job = fake_job.FakeLegacyJob(self, job_id, run_job)
            out_job.submit()
        return out_job
