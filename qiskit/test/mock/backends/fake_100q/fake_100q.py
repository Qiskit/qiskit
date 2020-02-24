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

"""
Fake 100 qubit device.
"""
import itertools
import os
import json
from datetime import datetime
from typing import Optional, List
import numpy as np

from qiskit.providers.models import (PulseBackendConfiguration,
                                     BackendProperties, PulseDefaults, BackendConfiguration, GateConfig, UchannelLO,
                                     Command)
from qiskit.providers.models.backendproperties import Nduv, Gate
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.test.mock.fake_backend import FakeBackend


class FakeOpenPulse100Q(FakeBackend):
    """A fake 100Q backend."""

    def __init__(self):
        """
        TODO: architecture
        """
        dirname = os.path.dirname(__file__)
        filename = "conf_100q.json"
        with open(os.path.join(dirname, filename), "r") as f_conf:
            conf = json.load(f_conf)

        configuration = PulseBackendConfiguration.from_dict(conf)
        configuration.backend_name = 'fake_openpulse_100q'
        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties."""
        dirname = os.path.dirname(__file__)
        filename = "props_100q.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)

    def defaults(self):
        """Returns a snapshot of device defaults."""
        dirname = os.path.dirname(__file__)
        filename = "defs_100q.json"
        with open(os.path.join(dirname, filename), "r") as f_defs:
            defs = json.load(f_defs)
        return PulseDefaults.from_dict(defs)


# TODO: move to utils or remove if not necessary
class FakeBackendBuilder(object):
    """FakeBackend builder.

    For example:

        builder = FakeBackendBuilder("tashkent", grid_size=10)
        path = os.path.dirname(os.path.abspath(__file__))
        builder.dump(path)
    """

    def __init__(self,
                 name: str,
                 grid_size: int,
                 version: Optional[str] = '0.0.0'):
        """

        Args:
            name:
            grid_size:
            version:
        """
        self.name = name
        self.grid_size = grid_size
        self.version = version
        self.now = datetime.now()
        self.basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
        self.n_qubits = grid_size**2

    def _generate_cmap(self) -> List[List[int]]:
        """Generate Almaden like connectivity map."""
        cmap = []
        for i in range(self.n_qubits):
            if i % self.grid_size != 0:
                cmap.append([i, i + 1])
            if i % self.grid_size < self.grid_size and i % 2 == 0:
                cmap.append([i, i + self.grid_size])

        return cmap

    def build_props(self) -> BackendProperties:
        """Build properties for backend."""
        qubits = []
        gates = []

        for i in range(self.n_qubits):
            # TODO: correct values
            qubits.append([
                Nduv(date=self.now, name='T1', unit='µs', value=113.3),
                Nduv(date=self.now, name='T2', unit='µs', value=150.2),
                Nduv(date=self.now, name='frequency', unit='GHz', value=4.8),
                Nduv(date=self.now, name='readout_error', unit='', value=0.04),
                Nduv(date=self.now, name='prob_meas0_prep1', unit='', value=0.08),
                Nduv(date=self.now, name='prob_meas1_prep0', unit='', value=0.02)
            ])

            for gate in self.basis_gates:
                # TODO: correct values
                parameters = [Nduv(date=self.now, name='gate_error', unit='', value=1.0),
                              Nduv(date=self.now, name='gate_length', unit='', value=0.)]
                # TODO: Gate constructor parameters should be List[Nduv]
                # TODO: handle CNOT
                gates.append(Gate(gate=gate, name="{0}_{1}".format(gate, i),
                                  qubits=[i], parameters=parameters))

        return BackendProperties(backend_name=self.name,
                                 backend_version=self.version,
                                 last_update_date=self.now,
                                 qubits=qubits,
                                 gates=gates,
                                 general=[])

    def build_conf(self) -> PulseBackendConfiguration:
        """Build configuration for backend."""
        # TODO: correct values
        meas_map = [list(range(self.n_qubits))]
        hamiltonian = {'h_str': [], 'description': "", 'qub': {}, 'vars': {}}
        conditional_latency = []
        acquisition_latency = []
        discriminators = []
        meas_kernels = []
        channel_bandwidth = []
        rep_times = []
        meas_level = []
        qubit_lo_range = []
        meas_lo_range = []
        u_channel_lo = []

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
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=self._generate_cmap(),
            n_registers=self.n_qubits,
            n_uchannels=self.n_qubits,
            u_channel_lo=u_channel_lo,
            meas_level=meas_level,
            qubit_lo_range=qubit_lo_range,
            meas_lo_range=meas_lo_range,
            dt=1.3333,
            dtm=10.5,
            rep_times=rep_times,
            meas_map=meas_map,
            channel_bandwidth=channel_bandwidth,
            meas_kernels=meas_kernels,
            discriminators=discriminators,
            acquisition_latency=acquisition_latency,
            conditional_latency=conditional_latency,
            hamiltonian=hamiltonian
        )

    def build_defaults(self) -> PulseDefaults:
        """Build backend defaults."""

        qubit_freq_est = np.linspace(4.9, 5.1, self.n_qubits).tolist()
        meas_freq_est = np.linspace(6.4, 6.6, self.n_qubits).tolist()
        buffer = 10
        pulse_library = [PulseLibraryItem(name='test_pulse_1', samples=[0.j, 0.1j]),
                         PulseLibraryItem(name='test_pulse_2', samples=[0.j, 0.1j, 1j]),
                         PulseLibraryItem(name='test_pulse_3',
                                          samples=[0.j, 0.1j, 1j, 0.5 + 0j]),
                         PulseLibraryItem(name='test_pulse_4',
                                          samples=7*[0.j, 0.1j, 1j, 0.5 + 0j])]

        measure_command_sequence = [PulseQobjInstruction(name='acquire', duration=10, t0=0,
                                                         qubits=range(self.n_qubits),
                                                         memory_slot=range(self.n_qubits))]
        measure_command_sequence += [PulseQobjInstruction(name='test_pulse_1',
                                                          ch='m{}'.format(i), t0=0)
                                     for i in range(self.n_qubits)]

        measure_command = Command(name='measure', qubits=range(self.n_qubits),
                                  sequence=measure_command_sequence)

        cmd_def = [measure_command]

        for i in range(self.n_qubits):
            cmd_def += [
                Command(name='u1', qubits=[i],
                        sequence=[PulseQobjInstruction(name='fc', ch='d{}'.format(i),
                                                       t0=0, phase='-P0')]),
                Command(name='u2', qubits=[i],
                        sequence=[PulseQobjInstruction(name='fc', ch='d{}'.format(i),
                                                       t0=0, phase='-P1'),
                                  PulseQobjInstruction(name='test_pulse_4',
                                                       ch='d{}'.format(i), t0=0),
                                  PulseQobjInstruction(name='fc', ch='d{}'.format(i),
                                                       t0=0, phase='-P0')]),
                Command(name='u3', qubits=[i],
                        sequence=[PulseQobjInstruction(name='test_pulse_3',
                                                       ch='d{}'.format(i), t0=0)])
            ]

        for connected_pair in self._generate_cmap():
            q1, q2 = connected_pair
            cmd_def += [
                Command(name='cx', qubits=[q1, q2],
                        sequence=[PulseQobjInstruction(name='test_pulse_1',
                                                       ch='d{}'.format(q1), t0=0),
                                  PulseQobjInstruction(name='test_pulse_2',
                                                       ch='u{}'.format(q1), t0=10),
                                  PulseQobjInstruction(name='test_pulse_1',
                                                       ch='d{}'.format(q2), t0=20),
                                  PulseQobjInstruction(name='fc', ch='d{}'.format(q2),
                                                       t0=20, phase=2.1)])
            ]

        return PulseDefaults(
            qubit_freq_est=qubit_freq_est,
            meas_freq_est=meas_freq_est,
            buffer=buffer,
            pulse_library=pulse_library,
            cmd_def=cmd_def
        )

    def build_backend(self):
        """Build backend."""
        pass

    def dump(self, folder: str):
        """Dumps backend configuration files to specifier folder."""
        with open('{0}/props_{1}.json'.format(folder, self.name), 'w') as f:
            json.dump(self.build_props().to_dict(), f, indent=4, sort_keys=True)

        with open('{0}/conf_{1}.json'.format(folder, self.name), 'w') as f:
            json.dump(self.build_conf().to_dict(), f, indent=4, sort_keys=True)

        with open('{0}/defs_{1}.json'.format(folder, self.name), 'w') as f:
            json.dump(self.build_defaults().to_dict(), f,
                      indent=4, sort_keys=True,
                      default=lambda o: '')
