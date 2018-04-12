# -*- coding: utf-8 -*-
# pylint: disable=missing-param-doc,missing-type-doc
#
# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Quantum Job class"""
import random
import string
import qiskit.backends as backends
from qiskit.unroll import Unroller, DagUnroller, JsonBackend
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit
from qiskit.qasm import Qasm
from qiskit import _openquantumcompiler as openquantumcompiler


class QuantumJob():
    """Creates a quantum circuit job
    """

    # TODO We need to create more tests for checking all possible inputs.
    # TODO Make this interface clearer -- circuits could be many things!
    def __init__(self, circuits, backend='local_qasm_simulator',
                 config=None, circuit_config=None, seed=None,
                 resources=None,
                 shots=1024, names=None,
                 do_compile=False, preformatted=False):
        """
        Args:
            circuits (QuantumCircuit|DagCircuit | list(QuantumCircuit|DagCircuit)):
                QuantumCircuit|DagCircuit or list of QuantumCircuit|DagCircuit.
                If preformatted=True, this is a raw qobj.
            backend (str): The backend to run the circuit on.
            circuit_config (dict): Circuit configuration.
            seed (int): The intial seed the simulatros use.
            resources (dict): Resource requirements of job.
            shots (int): the number of shots
            names (str or list(str)): names/ids for circuits
            do_compile (boolean): compile flag.
            preformatted (bool): the objects in circuits are already compiled
                and formatted (qasm for online, json for local). If true the
                parameters "names" and "circuit_config" must also be defined
                of the same length as "circuits".
        """
        resources = resources or {'max_credits': 10, 'wait': 5, 'timeout': 120}
        if isinstance(circuits, list):
            self.circuits = circuits
        else:
            self.circuits = [circuits]
        if names is None:
            self.names = []
            for _ in range(len(self.circuits)):
                self.names.append(self._generate_job_id(length=10))
        elif isinstance(names, list):
            self.names = names
        else:
            self.names = [names]

        self.timeout = resources['timeout']
        self.wait = resources['wait']
        # check whether circuits have already been compiled
        # and formatted for backend.
        if preformatted:
            # circuits is actually a qobj...validate (not ideal but convenient)
            self.qobj = circuits
        else:
            self.qobj = self._create_qobj(circuits, config, circuit_config, backend,
                                          seed, resources, shots, do_compile)
        self.backend = self.qobj['config']['backend']
        self.resources = resources
        self.seed = seed
        self.result = None

    def _create_qobj(self, circuits, config, circuit_config, backend, seed,
                     resources, shots, do_compile):
        # local and remote backends currently need different
        # compilied circuit formats
        formatted_circuits = []
        if do_compile:
            for circuit in circuits:
                formatted_circuits.append(None)
        else:
            for circuit in self.circuits:
                formatted_circuits.append(
                    openquantumcompiler.dag2json(circuit))
        required_config = {
            'max_credits': resources['max_credits'],
            'shots': shots,
            'backend': backend
        }
        """
        else:
            if backend in backends.local_backends():
                for circuit in self.circuits:
                    basis = ['u1', 'u2', 'u3', 'cx', 'id']
                    unroller = Unroller
                    # TODO: No instanceof here! Refactor this class
                    if isinstance(circuit, DAGCircuit):
                        unroller = DagUnroller
                    elif isinstance(circuit, QuantumCircuit):
                        # TODO: We should remove this code path (it's redundant and slow)
                        circuit = Qasm(data=circuit.qasm()).parse()
                    unroller_instance = unroller(circuit, JsonBackend(basis))
                    compiled_circuit = unroller_instance.execute()
                    formatted_circuits.append(compiled_circuit)

            else:
                for circuit in self.circuits:
                    formatted_circuits.append(circuit.qasm(qeflag=True))
                    """
        # merge dicts, second overrides first. Used if user supplies a
        # circuit_config formatted config for the top level for backends which
        # use the top level config for backend specific defaults.
        if config is None:
            config = required_config
        else:
            config = {**config, **required_config}
        # create circuit component of qobj
        circuit_records = []
        if circuit_config is None:
            circuit_config = {'coupling_map': None,
                              'basis_gates': 'u1,u2,u3,cx,id',
                              'layout': None,
                              'seed': seed}
            circuit_config = [circuit_config] * len(self.circuits)

        for circuit, fcircuit, name, config in zip(self.circuits,
                                                   formatted_circuits,
                                                   self.names,
                                                   circuit_config):
            record = {
                'name': name,
                'compiled_circuit': None if do_compile else fcircuit,
                'compiled_circuit_qasm': None if do_compile else fcircuit,
                'circuit': circuit,
                'config': config
            }
            circuit_records.append(record)

        return {'id': self._generate_job_id(length=10),
                'config': config,
                'circuits': circuit_records}

    def _generate_job_id(self, length=10):
        return ''.join([random.choice(
            string.ascii_letters + string.digits) for i in range(length)])
