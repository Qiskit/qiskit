# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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

import logging

from qiskit import __version__ as terra_version
from qiskit.qobj import RunConfig
from qiskit.aqua.utils import compile_and_run_circuits
from qiskit.aqua.utils.backend_utils import (is_aer_provider,
                                             is_ibmq_provider,
                                             is_statevector_backend,
                                             is_simulator_backend,
                                             is_local_backend)

logger = logging.getLogger(__name__)


class QuantumInstance:
    """Quantum Backend including execution setting."""

    BACKEND_CONFIG = ['basis_gates', 'coupling_map']
    COMPILE_CONFIG = ['pass_manager', 'initial_layout', 'seed_mapper', 'qobj_id']
    RUN_CONFIG = ['shots', 'max_credits', 'memory', 'seed']
    QJOB_CONFIG = ['timeout', 'wait']
    NOISE_CONFIG = ['noise_model']

    #  https://github.com/Qiskit/qiskit-aer/blob/master/qiskit/providers/aer/backends/qasm_simulator.py
    BACKEND_OPTIONS_QASM_ONLY = ["statevector_sample_measure_opt", "max_parallel_shots"]
    BACKEND_OPTIONS = ["initial_statevector", "chop_threshold", "max_parallel_threads",
                       "max_parallel_experiments", "statevector_parallel_threshold",
                       "statevector_hpc_gate_opt"] + BACKEND_OPTIONS_QASM_ONLY

    def __init__(self, backend, run_config=None, initial_layout=None, pass_manager=None, seed_mapper=None,
                 backend_options=None, noise_model=None, timeout=None, wait=5, circuit_cache=None,
                 skip_qobj_validation=False):
        """Constructor.

        Args:
            backend (BaseBackend): instance of selected backend
            run_config (RunConfig): the run config see https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/qobj/run_config.py
            initial_layout (dict): initial layout of qubits in mapping
            pass_manager (PassManager): pass manager to handle how to compile the circuits
            seed_mapper (int): the random seed for circuit mapper
            backend_options (dict): all config setting for backend
            noise_model (qiskit.provider.aer.noise.noise_model.NoiseModel): noise model for simulator
            timeout (float or None): seconds to wait for job. If None, wait indefinitely.
            wait (float): seconds between queries to result
            circuit_cache (CircuitCache): A CircuitCache to use when calling compile_and_run_circuits
            skip_qobj_validation (bool): Bypass Qobj validation to decrease submission time
        """
        self._backend = backend
        # setup run config
        if run_config is None:
            run_config = RunConfig(shots=1024, max_credits=10, memory=False)

        if getattr(run_config, 'shots', None) is not None:
            if self.is_statevector and run_config.shots == 1:
                logger.info("statevector backend only works with shot=1, change "
                            "shots from {} to 1.".format(run_config.shots))
                run_config.shots = 1

        if getattr(run_config, 'memory', None) is not None:
            if not self.is_simulator and run_config.memory is True:
                logger.info("The memory flag only supports simulator rather than real device. "
                            "Change it to from {} to False.".format(run_config.memory))
                run_config.memory = False
        self._run_config = run_config

        # setup backend config
        coupling_map = getattr(backend.configuration(), 'coupling_map', None)
        # TODO: basis gates will be [str] rather than comma-separated str
        basis_gates = backend.configuration().basis_gates
        if isinstance(basis_gates, str):
            basis_gates = basis_gates.split(',')

        self._backend_config = {
            'basis_gates': basis_gates,
            'coupling_map': coupling_map
        }

        # setup noise config
        noise_config = None
        if noise_model is not None:
            if is_aer_provider(self._backend):
                if not self.is_statevector:
                    noise_config = noise_model
                else:
                    logger.info("The noise model can be only used with Aer qasm simulator. "
                                "Change it to None.")
            else:
                logger.info("The noise model can be only used with Qiskit Aer. "
                            "Please install it.")
        self._noise_config = {} if noise_config is None else {'noise_model': noise_config}

        # setup compile config
        self._compile_config = {
            'pass_manager': pass_manager,
            'initial_layout': initial_layout,
            'seed_mapper': seed_mapper,
            'qobj_id': None
        }

        # setup job config
        self._qjob_config = {'timeout': timeout} if self.is_local \
            else {'timeout': timeout, 'wait': wait}

        # setup backend options for run
        self._backend_options = {}
        if is_ibmq_provider(self._backend):
            logger.info("backend_options can not used with the backends in IBMQ provider.")
        else:
            self._backend_options = {} if backend_options is None \
                else {'backend_options': backend_options}

        self._shared_circuits = False
        self._circuit_summary = False
        self._circuit_cache = circuit_cache
        self._skip_qobj_validation = skip_qobj_validation

        logger.info(self)

    def __str__(self):
        """Overload string.

        Retruns:
            str: the info of the object.
        """
        info = "\nQiskit Terra version: {}\n".format(terra_version)
        info += "Backend: '{} ({})', with following setting:\n{}\n{}\n{}\n{}\n{}\n{}".format(
            self.backend_name, self._backend.provider(), self._backend_config, self._compile_config,
            self._run_config, self._qjob_config, self._backend_options, self._noise_config)
        return info

    def execute(self, circuits, **kwargs):
        """
        A wrapper to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute

        Returns:
            Result: Result object
        """
        result = compile_and_run_circuits(circuits, self._backend, self._backend_config,
                                          self._compile_config, self._run_config,
                                          self._qjob_config,
                                          backend_options=self._backend_options,
                                          noise_config=self._noise_config,
                                          show_circuit_summary=self._circuit_summary,
                                          has_shared_circuits=self._shared_circuits,
                                          circuit_cache=self._circuit_cache,
                                          skip_qobj_validation=self._skip_qobj_validation, **kwargs)
        if self._circuit_summary:
            self._circuit_summary = False

        return result

    def set_config(self, **kwargs):
        """Set configurations for the quantum instance."""
        for k, v in kwargs.items():
            if k in QuantumInstance.RUN_CONFIG:
                setattr(self._run_config, k, v)
            elif k in QuantumInstance.QJOB_CONFIG:
                self._qjob_config[k] = v
            elif k in QuantumInstance.COMPILE_CONFIG:
                self._compile_config[k] = v
            elif k in QuantumInstance.BACKEND_CONFIG:
                self._backend_config[k] = v
            elif k in QuantumInstance.BACKEND_OPTIONS:
                if is_ibmq_provider(self._backend):
                    logger.info("backend_options can not used with the backends in IBMQ provider.")
                else:
                    if k in QuantumInstance.BACKEND_OPTIONS_QASM_ONLY and self.is_statevector:
                        logger.info("'{}' is only applicable for qasm simulator but "
                                    "statevector simulator is used. Skip the setting.")
                    else:
                        if 'backend_options' not in self._backend_options:
                            self._backend_options['backend_options'] = {}
                        self._backend_options['backend_options'][k] = v
            elif k in QuantumInstance.NOISE_CONFIG:
                self._noise_config[k] = v
            else:
                raise ValueError("unknown setting for the key ({}).".format(k))

    @property
    def qjob_config(self):
        """Getter of qjob_config."""
        return self._qjob_config

    @property
    def backend_config(self):
        """Getter of backend_config."""
        return self._backend_config

    @property
    def compile_config(self):
        """Getter of compile_config."""
        return self._compile_config

    @property
    def run_config(self):
        """Getter of run_config."""
        return self._run_config

    @property
    def noise_config(self):
        """Getter of noise_config."""
        return self._noise_config

    @property
    def backend_options(self):
        """Getter of backend_options."""
        return self._backend_options

    @property
    def shared_circuits(self):
        """Getter of shared_circuits."""
        return self._shared_circuits

    @shared_circuits.setter
    def shared_circuits(self, new_value):
        self._shared_circuits = new_value

    @property
    def circuit_summary(self):
        """Getter of circuit summary."""
        return self._circuit_summary

    @circuit_summary.setter
    def circuit_summary(self, new_value):
        self._circuit_summary = new_value

    @property
    def backend(self):
        """Return BaseBackend backend object."""
        return self._backend

    @property
    def backend_name(self):
        """Return backend name."""
        return self._backend.name()

    @property
    def is_statevector(self):
        """Return True if backend is a statevector-type simulator."""
        return is_statevector_backend(self._backend)

    @property
    def is_simulator(self):
        """Return True if backend is a simulator."""
        return is_simulator_backend(self._backend)

    @property
    def is_local(self):
        """Return True if backend is a local backend."""
        return is_local_backend(self._backend)

    @property
    def circuit_cache(self):
        return self._circuit_cache

    @property
    def has_circuit_caching(self):
        return self._circuit_cache is not None

    @property
    def skip_qobj_validation(self):
        return self._skip_qobj_validation

    @skip_qobj_validation.setter
    def skip_qobj_validation(self, new_value):
        self._skip_qobj_validation = new_value
