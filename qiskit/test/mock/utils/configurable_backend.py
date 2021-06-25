# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Configurable backend."""
import itertools
from datetime import datetime
from typing import Optional, List, Union

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.providers.models import (
    PulseBackendConfiguration,
    BackendProperties,
    PulseDefaults,
    Command,
    UchannelLO,
)
from qiskit.providers.models.backendproperties import Nduv, Gate
from qiskit.qobj import PulseQobjInstruction
from qiskit.test.mock.fake_backend import FakeBackend


class ConfigurableFakeBackend(FakeBackend):
    """Configurable backend."""

    def __init__(
        self,
        name: str,
        n_qubits: int,
        version: Optional[str] = None,
        coupling_map: Optional[List[List[int]]] = None,
        basis_gates: Optional[List[str]] = None,
        qubit_t1: Optional[Union[float, List[float]]] = None,
        qubit_t2: Optional[Union[float, List[float]]] = None,
        qubit_frequency: Optional[Union[float, List[float]]] = None,
        qubit_readout_error: Optional[Union[float, List[float]]] = None,
        single_qubit_gates: Optional[List[str]] = None,
        dt: Optional[float] = None,
        std: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Creates backend based on provided configuration.

        Args:
            name: Name of the backend.
            n_qubits: Number of qubits in the backend.
            version: Version of the fake backend.
            coupling_map: Coupling map.
            basis_gates: Basis gates of the backend.
            qubit_t1: Longitudinal coherence times.
            qubit_t2: Transverse coherence times.
            qubit_frequency: Frequency of qubits.
            qubit_readout_error: Readout error of qubits.
            single_qubit_gates: List of single qubit gates for backend properties.
            dt: Discretization of the input time sequences.
            std: Standard deviation of the generated distributions.
            seed: Random seed.
        """
        np.random.seed(seed)

        if version is None:
            version = "0.0.0"

        if basis_gates is None:
            basis_gates = ["id", "u1", "u2", "u3", "cx"]

        if std is None:
            std = 0.01

        if not isinstance(qubit_t1, list):
            qubit_t1 = np.random.normal(loc=qubit_t1 or 113.0, scale=std, size=n_qubits).tolist()

        if not isinstance(qubit_t2, list):
            qubit_t2 = np.random.normal(loc=qubit_t1 or 150.2, scale=std, size=n_qubits).tolist()

        if not isinstance(qubit_frequency, list):
            qubit_frequency = np.random.normal(
                loc=qubit_frequency or 4.8, scale=std, size=n_qubits
            ).tolist()

        if not isinstance(qubit_readout_error, list):
            qubit_readout_error = np.random.normal(
                loc=qubit_readout_error or 0.04, scale=std, size=n_qubits
            ).tolist()

        if single_qubit_gates is None:
            single_qubit_gates = ["id", "u1", "u2", "u3"]

        if dt is None:
            dt = 1.33

        self.name = name
        self.version = version
        self.basis_gates = basis_gates
        self.qubit_t1 = qubit_t1
        self.qubit_t2 = qubit_t2
        self.qubit_frequency = qubit_frequency
        self.qubit_readout_error = qubit_readout_error
        self.n_qubits = n_qubits
        self.single_qubit_gates = single_qubit_gates
        self.now = datetime.now()
        self.dt = dt
        self.std = std

        if coupling_map is None:
            coupling_map = self._generate_cmap()
        self.coupling_map = coupling_map

        configuration = self._build_conf()
        self._configuration = configuration
        self._defaults = self._build_defaults()
        self._properties = self._build_props()

        super().__init__(configuration)

    def defaults(self):
        """Return backend defaults."""
        return self._defaults

    def properties(self):
        """Return backend properties"""
        return self._properties

    def _generate_cmap(self) -> List[List[int]]:
        """Generate default grid-like coupling map."""
        cmap = []
        grid_size = int(np.ceil(np.sqrt(self.n_qubits)))

        for row in range(grid_size):
            for column in range(grid_size):
                if column + 1 < grid_size and column + row * grid_size + 1 < self.n_qubits:
                    qubit1 = column + row * grid_size
                    qubit2 = qubit1 + 1
                    cmap.append([qubit1, qubit2])
                if row + 1 < grid_size and column + (row + 1) * grid_size < self.n_qubits:
                    qubit1 = column + row * grid_size
                    qubit2 = qubit1 + grid_size
                    cmap.append([qubit1, qubit2])

        return cmap

    def _build_props(self) -> BackendProperties:
        """Build properties for backend."""
        qubits = []
        gates = []

        for (qubit_t1, qubit_t2, freq, read_err) in zip(
            self.qubit_t1, self.qubit_t2, self.qubit_frequency, self.qubit_readout_error
        ):
            qubits.append(
                [
                    Nduv(date=self.now, name="T1", unit="µs", value=qubit_t1),
                    Nduv(date=self.now, name="T2", unit="µs", value=qubit_t2),
                    Nduv(date=self.now, name="frequency", unit="GHz", value=freq),
                    Nduv(date=self.now, name="readout_error", unit="", value=read_err),
                ]
            )

        for gate in self.basis_gates:
            parameters = [
                Nduv(date=self.now, name="gate_error", unit="", value=0.01),
                Nduv(date=self.now, name="gate_length", unit="ns", value=4 * self.dt),
            ]

            if gate in self.single_qubit_gates:
                for i in range(self.n_qubits):
                    gates.append(
                        Gate(
                            gate=gate,
                            name=f"{gate}_{i}",
                            qubits=[i],
                            parameters=parameters,
                        )
                    )
            elif gate == "cx":
                for (qubit1, qubit2) in list(itertools.combinations(range(self.n_qubits), 2)):
                    gates.append(
                        Gate(
                            gate=gate,
                            name=f"{gate}{qubit1}_{qubit2}",
                            qubits=[qubit1, qubit2],
                            parameters=parameters,
                        )
                    )
            else:
                raise QiskitError(
                    "{gate} is not supported by fake backend builder." "".format(gate=gate)
                )

        return BackendProperties(
            backend_name=self.name,
            backend_version=self.version,
            last_update_date=self.now,
            qubits=qubits,
            gates=gates,
            general=[],
        )

    def _build_conf(self) -> PulseBackendConfiguration:
        """Build configuration for backend."""
        h_str = [
            ",".join([f"_SUM[i,0,{self.n_qubits}", "wq{i}/2*(I{i}-Z{i})]"]),
            ",".join([f"_SUM[i,0,{self.n_qubits}", "omegad{i}*X{i}||D{i}]"]),
        ]
        variables = []
        for (qubit1, qubit2) in self.coupling_map:
            h_str += [
                "jq{q1}q{q2}*Sp{q1}*Sm{q2}".format(q1=qubit1, q2=qubit2),
                "jq{q1}q{q2}*Sm{q1}*Sp{q2}".format(q1=qubit1, q2=qubit2),
            ]

            variables.append((f"jq{qubit1}q{qubit2}", 0))
        for i, (qubit1, qubit2) in enumerate(self.coupling_map):
            h_str.append(f"omegad{qubit1}*X{qubit2}||U{i}")
        for i in range(self.n_qubits):
            variables += [(f"omegad{i}", 0), (f"wq{i}", 0)]
        hamiltonian = {
            "h_str": h_str,
            "description": f"Hamiltonian description for {self.n_qubits} qubits backend.",
            "qub": {i: 2 for i in range(self.n_qubits)},
            "vars": dict(variables),
        }

        meas_map = [list(range(self.n_qubits))]
        qubit_lo_range = [[freq - 0.5, freq + 0.5] for freq in self.qubit_frequency]
        meas_lo_range = [[6.5, 7.5] for _ in range(self.n_qubits)]
        u_channel_lo = [[UchannelLO(q=i, scale=1.0 + 0.0j)] for i in range(len(self.coupling_map))]

        return PulseBackendConfiguration(
            backend_name=self.name,
            backend_version=self.version,
            n_qubits=self.n_qubits,
            meas_levels=[0, 1, 2],
            basis_gates=self.basis_gates,
            simulator=False,
            local=True,
            conditional=True,
            open_pulse=True,
            memory=False,
            max_shots=65536,
            gates=[],
            coupling_map=self.coupling_map,
            n_registers=self.n_qubits,
            n_uchannels=self.n_qubits,
            u_channel_lo=u_channel_lo,
            meas_level=[1, 2],
            qubit_lo_range=qubit_lo_range,
            meas_lo_range=meas_lo_range,
            dt=self.dt,
            dtm=10.5,
            rep_times=[1000],
            meas_map=meas_map,
            channel_bandwidth=[],
            meas_kernels=["kernel1"],
            discriminators=["max_1Q_fidelity"],
            acquisition_latency=[],
            conditional_latency=[],
            hamiltonian=hamiltonian,
        )

    def _build_defaults(self) -> PulseDefaults:
        """Build backend defaults."""

        qubit_freq_est = self.qubit_frequency
        meas_freq_est = np.linspace(6.4, 6.6, self.n_qubits).tolist()
        pulse_library = [
            {"name": "test_pulse_1", "samples": [[0.0, 0.0], [0.0, 0.1]]},
            {"name": "test_pulse_2", "samples": [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]},
            {"name": "test_pulse_3", "samples": [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.5, 0.0]]},
            {
                "name": "test_pulse_4",
                "samples": 7 * [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.5, 0.0]],
            },
        ]

        measure_command_sequence = [
            PulseQobjInstruction(
                name="acquire",
                duration=10,
                t0=0,
                qubits=list(range(self.n_qubits)),
                memory_slot=list(range(self.n_qubits)),
            ).to_dict()
        ]
        measure_command_sequence += [
            PulseQobjInstruction(name="test_pulse_1", ch=f"m{i}", t0=0).to_dict()
            for i in range(self.n_qubits)
        ]

        measure_command = Command.from_dict(
            {
                "name": "measure",
                "qubits": list(range(self.n_qubits)),
                "sequence": measure_command_sequence,
            }
        ).to_dict()

        cmd_def = [measure_command]

        for gate in self.single_qubit_gates:
            for i in range(self.n_qubits):
                cmd_def.append(
                    Command.from_dict(
                        {
                            "name": gate,
                            "qubits": [i],
                            "sequence": [
                                PulseQobjInstruction(
                                    name="fc", ch=f"d{i}", t0=0, phase="-P0"
                                ).to_dict(),
                                PulseQobjInstruction(
                                    name="test_pulse_3", ch=f"d{i}", t0=0
                                ).to_dict(),
                            ],
                        }
                    ).to_dict()
                )

        for qubit1, qubit2 in self.coupling_map:
            cmd_def += [
                Command.from_dict(
                    {
                        "name": "cx",
                        "qubits": [qubit1, qubit2],
                        "sequence": [
                            PulseQobjInstruction(
                                name="test_pulse_1", ch=f"d{qubit1}", t0=0
                            ).to_dict(),
                            PulseQobjInstruction(
                                name="test_pulse_2", ch=f"u{qubit1}", t0=10
                            ).to_dict(),
                            PulseQobjInstruction(
                                name="test_pulse_1", ch=f"d{qubit2}", t0=20
                            ).to_dict(),
                            PulseQobjInstruction(
                                name="fc", ch=f"d{qubit2}", t0=20, phase=2.1
                            ).to_dict(),
                        ],
                    }
                ).to_dict()
            ]

        return PulseDefaults.from_dict(
            {
                "qubit_freq_est": qubit_freq_est,
                "meas_freq_est": meas_freq_est,
                "buffer": 0,
                "pulse_library": pulse_library,
                "cmd_def": cmd_def,
            }
        )
