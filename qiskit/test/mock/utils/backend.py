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
Fake backend generation.
"""
import itertools
import json
from datetime import datetime
from typing import Optional, List, Type

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.providers.models import (PulseBackendConfiguration,
                                     BackendProperties, PulseDefaults, GateConfig, Command)
from qiskit.providers.models.backendproperties import Nduv, Gate
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.test.mock.fake_backend import FakeBackend


class FakeBackendBuilder(object):
    """FakeBackend builder.

    For example:
        builder = FakeBackendBuilder("Tashkent", n_qubits=100)
        path = os.path.dirname(os.path.abspath(__file__))
        builder.dump(path)
    """

    _SINGLE_QUBIT_GATES = ['id', 'u1', 'u2', 'u3']
    _DEFAULT_BASIS_GATES = ['id', 'u1', 'u2', 'u3', 'cx']
    _DEFAULT_T1 = 113.3
    _DEFAULT_T2 = 150.2
    _DEFAULT_QUBIT_FREQUENCY = 4.8
    _DEFAULT_QUBIT_READOUT_ERROR = 0.04

    def __init__(self,
                 name: str,
                 n_qubits: int,
                 version: Optional[str] = '0.0.0',
                 coupling_map: Optional[List[List[int]]] = None,
                 basis_gates: Optional[List[str]] = None,
                 qubit_t1: Optional[float] = None,
                 qubit_t2: Optional[float] = None,
                 qubit_frequency: Optional[float] = None,
                 qubit_readout_error: Optional[float] = None
                 ):
        """

        Args:
            name:
            version:
        """
        self.name = name
        self.version = version
        self.now = datetime.now()
        self.basis_gates = basis_gates if basis_gates else self._DEFAULT_BASIS_GATES
        self.n_qubits = n_qubits

        self.coupling_map = coupling_map

        self.qubit_t1 = qubit_t1 if qubit_t1 else self._DEFAULT_T1
        self.qubit_t2 = qubit_t2 if qubit_t2 else self._DEFAULT_T2
        self.qubit_frequency = qubit_frequency if qubit_frequency \
            else self._DEFAULT_QUBIT_FREQUENCY
        self.qubit_readout_error = qubit_readout_error if qubit_readout_error \
            else self._DEFAULT_QUBIT_READOUT_ERROR

    @property
    def cmap(self):
        return self.coupling_map if self.coupling_map else self._generate_cmap()

    def _generate_cmap(self) -> List[List[int]]:
        """Generate Almaden like coupling map."""
        cmap = []
        grid_size = int(np.ceil(np.sqrt(self.n_qubits)))
        for i in range(self.n_qubits):
            if i % grid_size != 0:
                cmap.append([i, i + 1])
            if i % grid_size < grid_size and i % 2 == 0:
                cmap.append([i, i + grid_size])

        self.coupling_map = cmap

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
            parameters = [Nduv(date=self.now, name='gate_error', unit='', value=1.0),
                          Nduv(date=self.now, name='gate_length', unit='', value=0.)]

            if gate in self._SINGLE_QUBIT_GATES:
                for i in range(self.n_qubits):
                    gates.append(Gate(gate=gate, name="{0}_{1}".format(gate, i),
                                      qubits=[i], parameters=parameters))
            elif gate == 'cx':
                for (q1, q2) in list(itertools.combinations(range(self.n_qubits), 2)):
                    gates.append(Gate(gate=gate,
                                      name="{gate}{q1}_{q2}".format(gate=gate, q1=q1, q2=q2),
                                      qubits=[q1, q2],
                                      parameters=parameters))
            else:
                raise QiskitError("{gate} is not supported by fake backend builder.".format(gate=gate))

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

        for connected_pair in self.cmap:
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


def fake_backend_generator(name: str,
                           n_qubits: int,
                           version: Optional[str] = '0.0.0',
                           coupling_map: Optional[List[List[int]]] = None,
                           basis_gates: Optional[List[str]] = None,
                           qubit_t1: Optional[float] = None,
                           qubit_t2: Optional[float] = None,
                           qubit_frequency: Optional[float] = None,
                           qubit_readout_error: Optional[float] = None) -> Type[FakeBackend]:
    """
    Example:
        FakeOpenPulse100Q = fake_backend_generator("fake_100_q_backend", 100)
        instance = FakeOpenPulse100Q()
    """

    builder = FakeBackendBuilder(name=name,
                                 n_qubits=n_qubits,
                                 version=version,
                                 coupling_map=coupling_map,
                                 basis_gates=basis_gates,
                                 qubit_t1=qubit_t1,
                                 qubit_t2=qubit_t2,
                                 qubit_frequency=qubit_frequency,
                                 qubit_readout_error=qubit_readout_error)

    def fake_init(self):
        configuration = builder.build_conf()
        super(FakeBackend, self).__init__(configuration)

    def properties(self) -> BackendProperties:
        return builder.build_props()

    def defaults(self) -> PulseDefaults:
        return builder.build_defaults()

    return type('FakeOpenPulse{}Q'.format(n_qubits),
                (FakeBackend,),
                {
                    '__init__': fake_init,
                    'backend_name': name,
                    'properties': properties,
                    'defaults': defaults
                })
