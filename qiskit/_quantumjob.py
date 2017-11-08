# -*- coding: utf-8 -*-
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
# ========================================================
"""Quantum Job class"""
from qiskit._qobj import QobjCircuitConfig, QobjCircuit, Qobj, QobjConfig
from . import backends
from . import _openquantumcompiler as openquantumcompiler
from ._util import random_string


class QuantumJob(object):
    """Creates a quantum circuit job

    Attributes:
       qobj (Qobj): describes circuits and configuration to run them
    """
    def __init__(self, qobj, resources=None):
        """

        Args:
            qobj (Qobj): describes circuits and configuration to run them
        """
        self.qobj = qobj
        self.backend = qobj.config.backend
        self.result = None
        resources = resources or {'max_credits':3, 'wait':5, 'timeout':120}
        self.timeout = resources['timeout']
        self.wait = resources['wait']

        return

    # TODO We need to create more tests for checking all possible inputs.
    @classmethod
    def from_circuits(cls,
                      circuits, backend='local_qasm_simulator',
                      circuit_config=None, seed=None,
                      resources=None, shots=1024, names=None,
                      do_compile=False, preformatted=False):
        """
        Args:
            circuits (QuantumCircuit | list(QuantumCircuit) | qobj): 
                QuantumCircuit or list of QuantumCircuit. If preformatted=True,
                this is a raw qobj.
            backend (str): The backend to run the circuit on.
            timeout (float): Timeout for job in seconds.
            seed (int): The intial seed the simulatros use.
            resources (dict): Resource requirements of job.
            shots (int): the number of shots
            circuit_type (str): "compiled_dag" or "uncompiled_dag" or
                "quantum_circuit"
            names (str | list(str)): names/ids for circuits
            preformatted (bool): the objects in circuits are already compiled
                and formatted (qasm for online, json for local). If true the
                parameters "names" and "circuit_config" must also be defined
                of the same length as "circuits".
        """
        # Preprocess the parameters.
        resources = resources or {'max_credits': 3, 'wait': 5, 'timeout': 120}
        if not isinstance(circuits, list):
            circuits = [circuits]

        if not names:
            names = [random_string(10) for _ in range(len(circuits))]
        elif not isinstance(names, list):
            names = [names]

        # check whether circuits have already been compiled
        # and formatted for backend.
        if preformatted:
            # circuits is actually a qobj...validate (not ideal but conventient)
            qobj = circuits
        else:
            qobj = cls._create_qobj(circuits, names, circuit_config, backend,
                                    seed, resources, shots, do_compile)

        return cls(qobj)

    @staticmethod
    def _create_qobj(circuits, names, circuit_config, backend, seed,
                     resources, shots, do_compile):
        # local and remote backends currently need different
        # compiled circuit formats
        formatted_circuits = []
        if do_compile:
            formatted_circuits.extend([None] * len(circuits))
        else:
            if backend in backends.local_backends():
                for circuit in circuits:
                    formatted_circuits.append(openquantumcompiler.dag2json(circuit))
            else:
                for circuit in circuits:
                    formatted_circuits.append(circuit.qasm(qeflag=True))

        # Create the Qobj without the circuits.
        qobj = Qobj(id_=random_string(10),
                    config=QobjConfig(backend=backend,
                                      max_credits=resources[
                                          'max_credits'],
                                      shots=shots),
                    circuits=[])

        # Create and add the QobjCircuits to the Qobj.
        if not circuit_config:
            qobj_circuit_config = QobjCircuitConfig(seed=seed)
            circuit_config = [qobj_circuit_config] * len(circuits)

        for circuit, fcircuit, name, config in zip(circuits,
                                                   formatted_circuits,
                                                   names,
                                                   circuit_config):
            qobj_circuit = QobjCircuit(name=name, config=config,
                                       compiled_circuit=None if do_compile else fcircuit,
                                       circuit=circuit,
                                       compiled_circuit_qasm=None if do_compile else fcircuit)
            qobj.circuits.append(qobj_circuit)

        return qobj
