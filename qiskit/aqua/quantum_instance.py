# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import time

from qiskit import __version__ as terra_version
from qiskit.assembler.run_config import RunConfig
from qiskit.transpiler import Layout

from .utils import (run_qobjs, compile_circuits, CircuitCache,
                    get_measured_qubits_from_qobj,
                    build_measurement_error_mitigation_fitter,
                    mitigate_measurement_error)
from .utils.backend_utils import (is_aer_provider,
                                  is_ibmq_provider,
                                  is_statevector_backend,
                                  is_simulator_backend,
                                  is_local_backend)

logger = logging.getLogger(__name__)


class QuantumInstance:
    """Quantum Backend including execution setting."""

    BACKEND_CONFIG = ['basis_gates', 'coupling_map']
    COMPILE_CONFIG = ['pass_manager', 'initial_layout', 'seed_transpiler']
    RUN_CONFIG = ['shots', 'max_credits', 'memory', 'seed']
    QJOB_CONFIG = ['timeout', 'wait']
    NOISE_CONFIG = ['noise_model']

    #  https://github.com/Qiskit/qiskit-aer/blob/master/qiskit/providers/aer/backends/qasm_simulator.py
    BACKEND_OPTIONS_QASM_ONLY = ["statevector_sample_measure_opt", "max_parallel_shots"]
    BACKEND_OPTIONS = ["initial_statevector", "chop_threshold", "max_parallel_threads",
                       "max_parallel_experiments", "statevector_parallel_threshold",
                       "statevector_hpc_gate_opt"] + BACKEND_OPTIONS_QASM_ONLY

    def __init__(self, backend, shots=1024, seed=None, max_credits=10,
                 basis_gates=None, coupling_map=None,
                 initial_layout=None, pass_manager=None, seed_transpiler=None,
                 backend_options=None, noise_model=None, timeout=None, wait=5,
                 circuit_caching=True, cache_file=None, skip_qobj_deepcopy=True,
                 skip_qobj_validation=True, measurement_error_mitigation_cls=None,
                 cals_matrix_refresh_period=30):
        """Constructor.

        Args:
            backend (BaseBackend): instance of selected backend
            shots (int, optional): number of repetitions of each circuit, for sampling
            seed (int, optional): random seed for simulators
            max_credits (int, optional): maximum credits to use
            basis_gates (list[str], optional): list of basis gate names supported by the
                                                target. Default: ['u1','u2','u3','cx','id']
            coupling_map (list[list]): coupling map (perhaps custom) to target in mapping
            initial_layout (dict, optional): initial layout of qubits in mapping
            pass_manager (PassManager, optional): pass manager to handle how to compile the circuits
            seed_transpiler (int, optional): the random seed for circuit mapper
            backend_options (dict, optional): all running options for backend, please refer to the provider.
            noise_model (qiskit.provider.aer.noise.noise_model.NoiseModel, optional): noise model for simulator
            timeout (float, optional): seconds to wait for job. If None, wait indefinitely.
            wait (float, optional): seconds between queries to result
            circuit_caching (bool, optional): USe CircuitCache when calling compile_and_run_circuits
            cache_file(str, optional): filename into which to store the cache as a pickle file
            skip_qobj_deepcopy (bool, optional): Reuses the same qobj object over and over to avoid deepcopying
            skip_qobj_validation (bool, optional): Bypass Qobj validation to decrease submission time
            measurement_error_mitigation_cls (callable, optional): the approach to mitigate measurement error,
                                                                CompleteMeasFitter or TensoredMeasFitter
            cals_matrix_refresh_period (int): how long to refresh the calibration matrix in measurement mitigation,
                                                  unit in minutes
        """
        self._backend = backend
        # setup run config
        run_config = RunConfig(shots=shots, max_credits=max_credits)
        if seed:
            run_config.seed = seed

        if getattr(run_config, 'shots', None) is not None:
            if self.is_statevector and run_config.shots != 1:
                logger.info("statevector backend only works with shot=1, change "
                            "shots from {} to 1.".format(run_config.shots))
                run_config.shots = 1

        self._run_config = run_config

        # setup backend config
        basis_gates = basis_gates or backend.configuration().basis_gates
        coupling_map = coupling_map or getattr(backend.configuration(),
                                               'coupling_map', None)
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
        if initial_layout is not None and not isinstance(initial_layout, Layout):
            initial_layout = Layout(initial_layout)
        self._compile_config = {
            'pass_manager': pass_manager,
            'initial_layout': initial_layout,
            'seed_transpiler': seed_transpiler
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
        self._circuit_cache = CircuitCache(skip_qobj_deepcopy=skip_qobj_deepcopy,
                                           cache_file=cache_file) if circuit_caching else None
        self._skip_qobj_validation = skip_qobj_validation

        self._measurement_error_mitigation_cls = None
        if self.is_statevector:
            if measurement_error_mitigation_cls is not None:
                logger.info("Measurement error mitigation does not work with statevector simulation, disable it.")
        else:
            self._measurement_error_mitigation_cls = measurement_error_mitigation_cls
        self._measurement_error_mitigation_fitter = None
        self._measurement_error_mitigation_method = 'least_squares'
        self._cals_matrix_refresh_period = cals_matrix_refresh_period
        self._prev_timestamp = 0

        if self._measurement_error_mitigation_cls is not None:
            logger.info("The measurement error mitigation is enable. "
                        "It will automatically submit an additional job to help calibrate the result of other jobs. "
                        "The current approach will submit a job with 2^N circuits to build the calibration matrix, "
                        "where N is the number of measured qubits. "
                        "Furthermore, Aqua will re-use the calibration matrix for {} minutes "
                        "and re-build it after that.".format(self._cals_matrix_refresh_period))

        logger.info(self)

    def __str__(self):
        """Overload string.

        Returns:
            str: the info of the object.
        """
        info = "\nQiskit Terra version: {}\n".format(terra_version)
        info += "Backend: '{} ({})', with following setting:\n{}\n{}\n{}\n{}\n{}\n{}".format(
            self.backend_name, self._backend.provider(), self._backend_config, self._compile_config,
            self._run_config, self._qjob_config, self._backend_options, self._noise_config)
        info += "\nMeasurement mitigation: {}".format(self._measurement_error_mitigation_cls)

        return info

    def execute(self, circuits, **kwargs):
        """
        A wrapper to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute

        Returns:
            Result: Result object
        """
        qobjs = compile_circuits(circuits, self._backend, self._backend_config, self._compile_config, self._run_config,
                                 show_circuit_summary=self._circuit_summary, circuit_cache=self._circuit_cache,
                                 **kwargs)

        if self._measurement_error_mitigation_cls is not None:
            if self.maybe_refresh_cals_matrix():
                logger.info("Building calibration matrix for measurement error mitigation.")
                qubit_list = get_measured_qubits_from_qobj(qobjs)
                self._measurement_error_mitigation_fitter = build_measurement_error_mitigation_fitter(qubit_list,
                                                                                          self._measurement_error_mitigation_cls,
                                                                                          self._backend,
                                                                                          self._backend_config,
                                                                                          self._compile_config,
                                                                                          self._run_config,
                                                                                          self._qjob_config,
                                                                                          self._backend_options,
                                                                                          self._noise_config)

        result = run_qobjs(qobjs, self._backend, self._qjob_config, self._backend_options, self._noise_config,
                           self._skip_qobj_validation)

        if self._measurement_error_mitigation_fitter is not None:
            result = mitigate_measurement_error(result, self._measurement_error_mitigation_fitter,
                                                self._measurement_error_mitigation_method)

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

    def maybe_refresh_cals_matrix(self):
        """
        Calculate the time difference from the query of last time.

        Returns:
            bool: whether or not refresh the cals_matrix
        """
        ret = False
        curr_timestamp = time.time()
        difference = int(curr_timestamp - self._prev_timestamp) / 60.0
        if difference > self._cals_matrix_refresh_period:
            self._prev_timestamp = curr_timestamp
            ret = True

        return ret

    @property
    def cals_matrix(self):
        cals_matrix = None
        if self._measurement_error_mitigation_fitter is not None:
            cals_matrix = self._measurement_error_mitigation_fitter.cal_matrix
        return cals_matrix