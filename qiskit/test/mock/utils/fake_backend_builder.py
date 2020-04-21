# -*- coding: utf-8 -*-

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

"""Fake backend generation."""
import itertools
from datetime import datetime
from typing import Optional, List

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.providers.models import (PulseBackendConfiguration,
                                     BackendProperties, PulseDefaults,
                                     Command, UchannelLO)
from qiskit.providers.models.backendproperties import Nduv, Gate
from qiskit.qobj import PulseQobjInstruction
from qiskit.test.mock.fake_backend import FakeBackend


class FakeBackendBuilder:
    """FakeBackend builder."""

    def __init__(self,
                 name: str,
                 n_qubits: int,
                 version: Optional[str] = None,
                 coupling_map: Optional[List[List[int]]] = None,
                 basis_gates: Optional[List[str]] = None,
                 qubit_t1: Optional[float] = None,
                 qubit_t2: Optional[float] = None,
                 qubit_frequency: Optional[float] = None,
                 qubit_readout_error: Optional[float] = None,
                 single_qubit_gates: Optional[List[str]] = None,
                 dt: Optional[float] = None):
        """Creates fake backend builder.

        Args:
            name (str): Name of the backend.
            n_qubits (int): Number of qubits in the backend.
            version (str, optional): Version of the fake backend.
            coupling_map (list, optional): Coupling map.
            basis_gates (list, optional): Basis gates of the backend.
            qubit_t1 (float, optional): Longitudinal coherence time.
            qubit_t2 (float, optional): Transverse coherence time.
            qubit_frequency (float, optional): Frequency of qubit.
            qubit_readout_error (float, optional): Readout error of qubit.
            single_qubit_gates (list, optional): List of single qubit gates for backend properties.
            dt (float, optional): Discretization of the input time sequences.
        """
        if version is None:
            version = '0.0.0'

        if basis_gates is None:
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']

        if qubit_t1 is None:
            qubit_t1 = 113.3

        if qubit_t2 is None:
            qubit_t2 = 150.2

        if qubit_frequency is None:
            qubit_frequency = 4.8

        if qubit_readout_error is None:
            qubit_readout_error = 0.04

        if single_qubit_gates is None:
            single_qubit_gates = ['id', 'u1', 'u2', 'u3']

        if dt is None:
            dt = 1.33e-9

        if coupling_map is None:
            coupling_map = self._generate_cmap()

        self.name = name
        self.version = version
        self.basis_gates = basis_gates
        self.qubit_t1 = qubit_t1
        self.qubit_t2 = qubit_t2
        self.qubit_frequency = qubit_frequency
        self.qubit_readout_error = qubit_readout_error
        self.n_qubits = n_qubits
        self.single_qubit_gates = single_qubit_gates
        self.coupling_map = coupling_map
        self.now = datetime.now()
        self.dt = dt  # pylint: disable=invalid-name

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

    def build_props(self) -> BackendProperties:
        """Build properties for backend."""
        qubits = []
        gates = []

        for i in range(self.n_qubits):
            qubits.append([
                Nduv(date=self.now, name='T1', unit='µs', value=self.qubit_t1),
                Nduv(date=self.now, name='T2', unit='µs', value=self.qubit_t2),
                Nduv(date=self.now, name='frequency', unit='GHz', value=self.qubit_frequency),
                Nduv(date=self.now, name='readout_error', unit='', value=self.qubit_readout_error)
            ])

        for gate in self.basis_gates:
            parameters = [Nduv(date=self.now, name='gate_error', unit='', value=0.01),
                          Nduv(date=self.now, name='gate_length', unit='second', value=4*self.dt)]

            if gate in self.single_qubit_gates:
                for i in range(self.n_qubits):
                    gates.append(Gate(gate=gate, name="{0}_{1}".format(gate, i),
                                      qubits=[i], parameters=parameters))
            elif gate == 'cx':
                for (qubit1, qubit2) in list(itertools.combinations(range(self.n_qubits), 2)):
                    gates.append(Gate(gate=gate,
                                      name="{gate}{q1}_{q2}".format(gate=gate,
                                                                    q1=qubit1,
                                                                    q2=qubit2),
                                      qubits=[qubit1, qubit2],
                                      parameters=parameters))
            else:
                raise QiskitError("{gate} is not supported by fake backend builder."
                                  "".format(gate=gate))

        return BackendProperties(backend_name=self.name,
                                 backend_version=self.version,
                                 last_update_date=self.now,
                                 qubits=qubits,
                                 gates=gates,
                                 general=[])

    def build_conf(self) -> PulseBackendConfiguration:
        """Build configuration for backend."""
        h_str = [
            ",".join(["_SUM[i,0,{n_qubits}".format(n_qubits=self.n_qubits),
                      "wq{i}/2*(I{i}-Z{i})]"]),
            ",".join(["_SUM[i,0,{n_qubits}".format(n_qubits=self.n_qubits),
                      "omegad{i}*X{i}||D{i}]"])
        ]
        variables = []
        for (qubit1, qubit2) in self.coupling_map:
            h_str += [
                "jq{q1}q{q2}*Sp{q1}*Sm{q2}".format(q1=qubit1, q2=qubit2),
                "jq{q1}q{q2}*Sm{q1}*Sp{q2}".format(q1=qubit1, q2=qubit2)
            ]

            variables.append(("jq{q1}q{q2}".format(q1=qubit1, q2=qubit2), 0))
        for i, (qubit1, qubit2) in enumerate(self.coupling_map):
            h_str.append("omegad{0}*X{1}||U{2}".format(qubit1, qubit2, i))
        for i in range(self.n_qubits):
            variables += [
                ("omegad{}".format(i), 0),
                ("wq{}".format(i), 0)
            ]
        hamiltonian = {
            'h_str': h_str,
            'description': 'Hamiltonian description for {} qubits backend.'.format(self.n_qubits),
            'qub': {i: 2 for i in range(self.n_qubits)},
            'vars': dict(variables)
        }

        meas_map = [list(range(self.n_qubits))]
        qubit_lo_range = [[self.qubit_frequency - .5, self.qubit_frequency + .5]
                          for _ in range(self.n_qubits)]
        meas_lo_range = [[6.5, 7.5] for _ in range(self.n_qubits)]
        u_channel_lo = [[UchannelLO(q=i, scale=1. + 0.j)] for i in range(len(self.coupling_map))]

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
            meas_kernels=['kernel1'],
            discriminators=['max_1Q_fidelity'],
            acquisition_latency=[],
            conditional_latency=[],
            hamiltonian=hamiltonian
        )

    def build_defaults(self) -> PulseDefaults:
        """Build backend defaults."""

        qubit_freq_est = [self.qubit_frequency] * self.n_qubits
        meas_freq_est = np.linspace(6.4, 6.6, self.n_qubits).tolist()
        pulse_library = [
            {
                'name': 'test_pulse_1',
                'samples': [[0.0, 0.0], [0.0, 0.1]]
            },
            {
                'name': 'test_pulse_2',
                'samples': [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]
            },
            {
                'name': 'test_pulse_3',
                'samples': [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.5, 0.0]]
            },
            {
                'name': 'test_pulse_4',
                'samples': 7 * [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.5, 0.0]]
            }
        ]

        measure_command_sequence = [PulseQobjInstruction(name='acquire', duration=10, t0=0,
                                                         qubits=list(range(self.n_qubits)),
                                                         memory_slot=list(range(self.n_qubits))
                                                         ).to_dict()]
        measure_command_sequence += [PulseQobjInstruction(name='test_pulse_1',
                                                          ch='m{}'.format(i), t0=0).to_dict()
                                     for i in range(self.n_qubits)]

        measure_command = Command.from_dict({
            'name': 'measure',
            'qubits': list(range(self.n_qubits)),
            'sequence': measure_command_sequence
        }).to_dict()

        cmd_def = [measure_command]

        for gate in self.single_qubit_gates:
            for i in range(self.n_qubits):
                cmd_def.append(Command.from_dict({
                    'name': gate,
                    'qubits': [i],
                    'sequence': [PulseQobjInstruction(name='fc', ch='d{}'.format(i),
                                                      t0=0, phase='-P0').to_dict(),
                                 PulseQobjInstruction(name='test_pulse_3',
                                                      ch='d{}'.format(i),
                                                      t0=0).to_dict()]
                }).to_dict())

        for qubit1, qubit2 in self.coupling_map:
            cmd_def += [
                Command.from_dict({
                    'name': 'cx',
                    'qubits': [qubit1, qubit2],
                    'sequence': [PulseQobjInstruction(name='test_pulse_1',
                                                      ch='d{}'.format(qubit1),
                                                      t0=0).to_dict(),
                                 PulseQobjInstruction(name='test_pulse_2',
                                                      ch='u{}'.format(qubit1),
                                                      t0=10).to_dict(),
                                 PulseQobjInstruction(name='test_pulse_1',
                                                      ch='d{}'.format(qubit2),
                                                      t0=20).to_dict(),
                                 PulseQobjInstruction(name='fc', ch='d{}'.format(qubit2),
                                                      t0=20, phase=2.1).to_dict()]
                }).to_dict()
            ]

        return PulseDefaults.from_dict({
            'qubit_freq_est': meas_freq_est,
            'meas_freq_est': qubit_freq_est,
            'buffer': 0,
            'pulse_library': pulse_library,
            'cmd_def': cmd_def
        })

    def build(self):
        """Builds fake backend with specified parameters."""
        backend = FakeBackend(self.build_conf())
        backend.defaults = self.build_defaults
        backend.properties = self.build_props
        return backend
