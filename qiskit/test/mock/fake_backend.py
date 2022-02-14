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
import json
import os

from qiskit import circuit
from qiskit.providers.models import BackendProperties, PulseBackendConfiguration, PulseDefaults
from qiskit.providers import BackendV1, BackendV2, BaseBackend
from qiskit.providers.options import Options
from qiskit import pulse
from qiskit.circuit.parameter import Parameter
from qiskit.transpiler import Target, InstructionProperties
from qiskit.exceptions import QiskitError
from qiskit.test.mock import fake_job
from qiskit.test.mock.utils.json_decoder import (
    decode_backend_configuration,
    decode_backend_properties,
    decode_pulse_defaults
)
from qiskit.test.mock.utils.backend_converter import (
    convert_to_target,
    qubit_properties_dict_from_properties
)
from qiskit.utils import optionals as _optionals
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
        if _optionals.HAS_AER:
            from qiskit.providers import aer

            return aer.QasmSimulator._default_options()
        else:
            return basicaer.QasmSimulatorPy._default_options()

    def run(self, run_input, **kwargs):
        """Main job in simulator"""
        circuits = run_input
        pulse_job = None
        if isinstance(circuits, (pulse.Schedule, pulse.ScheduleBlock)):
            pulse_job = True
        elif isinstance(circuits, circuit.QuantumCircuit):
            pulse_job = False
        elif isinstance(circuits, list):
            if circuits:
                if all(isinstance(x, (pulse.Schedule, pulse.ScheduleBlock)) for x in circuits):
                    pulse_job = True
                elif all(isinstance(x, circuit.QuantumCircuit) for x in circuits):
                    pulse_job = False
        if pulse_job is None:
            raise QiskitError(
                "Invalid input object %s, must be either a "
                "QuantumCircuit, Schedule, or a list of either" % circuits
            )
        if _optionals.HAS_AER:
            from qiskit.providers import aer

            if pulse_job:
                from qiskit.providers.aer.pulse import PulseSystemModel

                system_model = PulseSystemModel.from_backend(self)
                sim = aer.Aer.get_backend("pulse_simulator")
                job = sim.run(circuits, system_model=system_model, **kwargs)
            else:
                sim = aer.Aer.get_backend("qasm_simulator")
                if self.properties():
                    from qiskit.providers.aer.noise import NoiseModel

                    noise_model = NoiseModel.from_backend(self, warnings=False)
                    job = sim.run(circuits, noise_model=noise_model, **kwargs)
                else:
                    job = sim.run(circuits, **kwargs)
        else:
            if pulse_job:
                raise QiskitError("Unable to run pulse schedules without qiskit-aer installed")
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
        if _optionals.HAS_AER:
            from qiskit.providers import aer

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
                raise QiskitError("Unable to run pulse schedules without qiskit-aer installed")
            warnings.warn("Aer not found using BasicAer and no noise", RuntimeWarning)

            def run_job():
                sim = basicaer.BasicAer.get_backend("qasm_simulator")
                return sim.run(qobj).result()

            job_id = uuid.uuid4()
            out_job = fake_job.FakeLegacyJob(self, job_id, run_job)
            out_job.submit()
        return out_job

class FakeBackendV2(BackendV2):
    """This is a dummy bakend just for resting purposes. the FakeBackendV2 builds on top of the BackendV2 base class."""

    def __init__(self):
        configuration = self._get_conf_from_json()
        super().__init__(
            provider=None,
            name=configuration.backend_name,
            description=configuration.description,
            online_date=configuration.online_date,
            backend_version=configuration.backend_version
        )
        self._properties = None
        self._qubit_properties = None
        self._defaults = None
        self._target = None

    def properties(self):
        """Returns a snapshot of device properties"""
        if not self._properties:
            self._set_props_from_json()
        return self._properties

    def defaults(self):
        """Returns a snapshot of device defaults"""
        if not self._defaults:
            self._set_defaults_from_json()
        return self._defaults

    def _get_conf_from_json(self):
        if not self.conf_filename:
            raise QiskitError("No configuration file has been defined")
        conf = self._load_json(self.conf_filename)
        decode_backend_configuration(conf)
        configuration = self._get_config_from_dict(conf)
        configuration.backend_name = self.backend_name
        return configuration

    def _get_config_from_dict(self, conf):
        return PulseBackendConfiguration.from_dict(conf)

    def _set_props_from_json(self):
        if not self.props_filename:
            raise QiskitError("No properties file has been defined")
        props = self._load_json(self.props_filename)
        decode_backend_properties(props)
        self._properties = BackendProperties.from_dict(props)

    def _set_defaults_from_json(self):
        if not self.props_filename:
            raise QiskitError("No properties file has been defined")
        defs = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs)
        self._defaults = PulseDefaults.from_dict(defs)

    def _load_json(self, filename):
        with open(os.path.join(self.dirname, filename)) as f_json:
            the_json = json.load(f_json)
        return the_json

    @property
    def target(self) -> Target:
        self._convert_to_target()
        return self._target

    def _convert_to_target(self) -> None:
        """Converts backend configuration, properties and defaults to Target object"""
        if not self._target:
            self._target = convert_to_target(
                configuration=self._configuration.to_dict(),
                properties=self._properties.to_dict() if self._properties else None,
                defaults=self._defaults.to_dict() if self._defaults else None,
            )

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, run_input, **options):
        raise NotImplementedError