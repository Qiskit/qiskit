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

from typing import List, Union

from qiskit import circuit
from qiskit.providers.models import BackendProperties
from qiskit.providers import BackendV1, BackendV2, BaseBackend, QubitProperties
from qiskit import pulse
from qiskit.exceptions import QiskitError
from qiskit.test.mock import fake_job
from qiskit.test.mock.utils.json_decoder import (
    decode_backend_configuration,
    decode_backend_properties,
    decode_pulse_defaults,
)
from qiskit.test.mock.utils.backend_converter import (
    convert_to_target,
    qubit_props_from_props,
)
from qiskit.utils import optionals as _optionals
from qiskit.providers import basicaer
from qiskit.transpiler import Target


class _Credentials:
    def __init__(self, token="123456", url="https://"):
        self.token = token
        self.url = url
        self.hub = "hub"
        self.group = "group"
        self.project = "project"


class FakeBackendV2(BackendV2):
    """This is a dummy bakend just for resting purposes.
    The FakeBackendV2 builds on top of the BackendV2 base class."""

    dirname = None
    conf_filename = None
    props_filename = None
    defs_filename = None
    backend_name = None

    def __init__(self):
        self._conf_dict = self._get_conf_dict_from_json()
        self._props_dict = self._set_props_dict_from_json()
        self._defs_dict = self._set_defs_dict_from_json()
        super().__init__(
            provider=None,
            name=self._conf_dict.get("backend_name"),
            description=self._conf_dict.get("description"),
            online_date=self._conf_dict.get("online_date"),
            backend_version=self._conf_dict.get("backend_version"),
        )
        self._target = convert_to_target(
            conf_dict=self._conf_dict,
            props_dict=self._props_dict,
            defs_dict=self._defs_dict,
        )
        self._qubit_properties = qubit_props_from_props(self._props_dict)

    def _get_conf_dict_from_json(self) -> dict:
        if not self.conf_filename:
            return None
        conf_dict = self._load_json(self.conf_filename)
        decode_backend_configuration(conf_dict)
        conf_dict["backend_name"] = self.backend_name
        return conf_dict

    def _set_props_dict_from_json(self) -> dict:
        if not self.props_filename:
            return None
        props_dict = self._load_json(self.props_filename)
        decode_backend_properties(props_dict)
        return props_dict

    def _set_defs_dict_from_json(self) -> dict:
        if not self.defs_filename:
            return None
        defs_dict = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs_dict)
        return defs_dict

    def _load_json(self, filename: str) -> dict:
        with open(os.path.join(self.dirname, filename)) as f_json:
            the_json = json.load(f_json)
        return the_json

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        if _optionals.HAS_AER:
            from qiskit.providers import aer

            return aer.QasmSimulator._default_options()
        else:
            return basicaer.QasmSimulatorPy._default_options()

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals
        Returns:
            dtm: The output signal timestep in seconds.
        """
        return self._conf_dict.get("dtm")

    @property
    def meas_map(self) -> List[List[int]]:
        """Return the grouping of measurements which are multiplexed
        This is required to be implemented if the backend supports Pulse
        scheduling.
        Returns:
            meas_map: The grouping of measurements which are multiplexed
        """
        return self._conf_dict.get("meas_map")

    def qubit_properties(
        self, qubit: Union[int, List[int]]
    ) -> Union[QubitProperties, List[QubitProperties]]:
        """Return QubitProperties for a given qubit.
        Args:
            qubit: The qubit to get the
                :class:`~qiskit.provider.QubitProperties` object for. This can
                be a single integer for 1 qubit or a list of qubits and a list
                of :class:`~qiskit.provider.QubitProperties` objects will be
                returned in the same order
        Returns:
            qubit_properties: The :class:`~qiskit.provider.QubitProperties` object
            for the specified qubit. If a list of qubits is provided a list will be
            returned. If properties are missing for a qubit this can be ``None``.
        """
        if isinstance(qubit, int):  # type: ignore[unreachable]
            return self._qubit_properties.get(qubit)
        if isinstance(qubit, List):
            return [self._qubit_properties.get(q) for q in qubit]
        return None

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
                raise QiskitError("Pulse simulation is currently not supported for V2 backends.")
            else:
                sim = aer.Aer.get_backend("qasm_simulator")
                if self._props_dict:
                    noise_model = self._get_noise_model_from_backend_v2(self, warnings=False)
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

    def _get_noise_model_from_backend_v2(
        self,
        gate_error=True,
        readout_error=True,
        thermal_relaxation=True,
        temperature=0,
        gate_lengths=None,
        gate_length_units="ns",
        standard_gates=None,
    ):
        """Build noise model from BackendV2.
        This is a temporary fix until Aer supports V2 backends.
        """

        from qiskit.circuit import Delay
        from qiskit.providers.exceptions import BackendPropertyError
        from qiskit.providers.aer.noise import NoiseModel
        from qiskit.providers.aer.noise.device.models import (
            _excited_population,
            basic_device_gate_errors,
            basic_device_readout_errors,
        )
        from qiskit.providers.aer.noise.passes import RelaxationNoisePass

        properties = BackendProperties.from_dict(self._props_dict)
        basis_gates = self.operation_names
        num_qubits = self.num_qubits
        dt = self.dt

        noise_model = NoiseModel(basis_gates=basis_gates)

        # Add single-qubit readout errors
        if readout_error:
            for qubits, error in basic_device_readout_errors(properties):
                noise_model.add_readout_error(error, qubits)

        # Add gate errors
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module="qiskit.providers.aer.noise.device.models",
            )
            gate_errors = basic_device_gate_errors(
                properties,
                gate_error=gate_error,
                thermal_relaxation=thermal_relaxation,
                gate_lengths=gate_lengths,
                gate_length_units=gate_length_units,
                temperature=temperature,
                standard_gates=standard_gates,
            )
        for name, qubits, error in gate_errors:
            noise_model.add_quantum_error(error, name, qubits)

        if thermal_relaxation:
            # Add delay errors via RelaxationNiose pass
            try:
                excited_state_populations = [
                    _excited_population(freq=properties.frequency(q), temperature=temperature)
                    for q in range(num_qubits)
                ]
            except BackendPropertyError:
                excited_state_populations = None
            try:
                delay_pass = RelaxationNoisePass(
                    t1s=[properties.t1(q) for q in range(num_qubits)],
                    t2s=[properties.t2(q) for q in range(num_qubits)],
                    dt=dt,
                    op_types=Delay,
                    excited_state_populations=excited_state_populations,
                )
                noise_model._custom_noise_passes.append(delay_pass)
            except BackendPropertyError:
                # Device does not have the required T1 or T2 information
                # in its properties
                pass

        return noise_model


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
