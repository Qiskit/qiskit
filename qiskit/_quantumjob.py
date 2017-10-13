"""Quantum Job class"""
import random
import string
from qiskit import _openquantumcompiler as openquantumcompiler
import qiskit.backends as backends

class QuantumJob():
    """Creates a quantum circuit job

    Attributes:
       qobj (dict): describes circuits and configuration to run them
    """

    # TODO We need to create more tests for checking all possible inputs.
    def __init__(self, circuits, backend='local_qasm_simulator',
                 circuit_config=None, seed=None,
                 resources={'max_credits':3, 'wait':5, 'timeout':120},
                 shots=1024, names=None,
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
            coupling_map (dict): A directed graph of coupling::

                {
                 control(int):
                     [
                         target1(int),
                         target2(int),
                         , ...
                    ],
                     ...
                }

                eg. {0: [2], 1: [2], 3: [2]}

            initial_layout (dict): A mapping of qubit to qubit::

                                  {
                                    ("q", strart(int)): ("q", final(int)),
                                    ...
                                  }
                                  eg.
                                  {
                                    ("q", 0): ("q", 0),
                                    ("q", 1): ("q", 1),
                                    ("q", 2): ("q", 2),
                                    ("q", 3): ("q", 3)
                                  }
            shots (int): the number of shots
            circuit_type (str): "compiled_dag" or "uncompiled_dag" or
                "quantum_circuit"
            names (str | list(str)): names/ids for circuits
            preformated (bool): the objects in circuits are already compiled
                and formatted (qasm for online, json for local). If true the
                parameters "names" and "circuit_config" must also be defined
                of the same length as "circuits".
        """
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
            # circuits is actually a qobj...validate (not ideal but conventient)
            self.qobj = circuits
        else:
            self.qobj = self._create_qobj(circuits, circuit_config, backend,
                                          seed, resources, shots, do_compile)
        self.backend = self.qobj['config']['backend']
        self.resources = resources
        self.seed = seed
        self.result = None

    def _create_qobj(self, circuits, circuit_config, backend, seed,
                     resources, shots, do_compile):
        # local and remote backends currently need different
        # compilied circuit formats
        formatted_circuits = []
        if do_compile:
            for circuit in circuits:
                formatted_circuits.append(None)
        else:
            if backend in backends.local_backends():
                for circuit in self.circuits:
                    formatted_circuits.append(openquantumcompiler.dag2json(circuit))
            else:
                for circuit in self.circuits:
                    formatted_circuits.append(circuit.qasm(qeflag=True))

        # create circuit component of qobj
        circuit_records = []
        if circuit_config is None:
            config = {'coupling_map': None,
                      'basis_gates': 'u1,u2,u3,cx,id',
                      'layout': None,
                      'seed': seed}
            circuit_config = [config] * len(self.circuits)

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
                'config': {
                    'max_credits': resources['max_credits'],
                    'shots': shots,
                    'backend': backend
                },
                'circuits': circuit_records}

    def _generate_job_id(self, length=10):
        return ''.join([random.choice(
            string.ascii_letters + string.digits) for i in range(length)])
