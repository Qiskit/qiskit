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

""" Quantum Instance module """

import copy
import logging
import time
import os
import warnings

from qiskit.assembler.run_config import RunConfig
from qiskit import compiler

from .aqua_error import AquaError
from .utils.backend_utils import (is_ibmq_provider,
                                  is_statevector_backend,
                                  is_simulator_backend,
                                  is_local_backend,
                                  is_aer_qasm,
                                  support_backend_options)
from .utils.circuit_utils import summarize_circuits

logger = logging.getLogger(__name__)


class QuantumInstance:
    """Quantum Backend including execution setting."""

    BACKEND_CONFIG = ['basis_gates', 'coupling_map']
    COMPILE_CONFIG = ['pass_manager', 'initial_layout', 'seed_transpiler', 'optimization_level']
    RUN_CONFIG = ['shots', 'max_credits', 'memory', 'seed_simulator']
    QJOB_CONFIG = ['timeout', 'wait']
    NOISE_CONFIG = ['noise_model']

    # https://github.com/Qiskit/qiskit-aer/blob/master/qiskit/providers/aer/backends/qasm_simulator.py
    BACKEND_OPTIONS_QASM_ONLY = ["statevector_sample_measure_opt", "max_parallel_shots"]
    BACKEND_OPTIONS = ["initial_statevector", "chop_threshold", "max_parallel_threads",
                       "max_parallel_experiments", "statevector_parallel_threshold",
                       "statevector_hpc_gate_opt"] + BACKEND_OPTIONS_QASM_ONLY

    def __init__(self, backend,
                 # run config
                 shots=1024, seed_simulator=None, max_credits=10,
                 # backend properties
                 basis_gates=None, coupling_map=None,
                 # transpile
                 initial_layout=None, pass_manager=None,
                 seed_transpiler=None, optimization_level=None,
                 # simulation
                 backend_options=None, noise_model=None,
                 # job
                 timeout=None, wait=5,
                 # others
                 circuit_caching=False, cache_file=None, skip_qobj_deepcopy=False,
                 skip_qobj_validation=True,
                 measurement_error_mitigation_cls=None, cals_matrix_refresh_period=30,
                 measurement_error_mitigation_shots=None,
                 job_callback=None):
        """Constructor.

        Args:
            backend (BaseBackend): instance of selected backend
            shots (int, optional): number of repetitions of each circuit, for sampling
            seed_simulator (int, optional): random seed for simulators
            max_credits (int, optional): maximum credits to use
            basis_gates (list[str], optional): list of basis gate names supported by the
                                               target. Default: ['u1','u2','u3','cx','id']
            coupling_map (CouplingMap or list[list]): coupling map (perhaps custom) to
                                                      target in mapping
            initial_layout (Layout or dict or list, optional): initial layout of qubits in mapping
            pass_manager (PassManager, optional): pass manager to handle how to compile the circuits
            seed_transpiler (int, optional): the random seed for circuit mapper
            optimization_level (int, optional): How much optimization to perform on the circuits.
                                                Higher levels generate more optimized circuits,
                                                at the expense of longer transpilation time.
            backend_options (dict, optional): all running options for backend, please refer
                                              to the provider.
            noise_model (qiskit.provider.aer.noise.noise_model.NoiseModel, optional): noise model
                                                                                      for simulator
            timeout (float, optional): seconds to wait for job. If None, wait indefinitely.
            wait (float, optional): seconds between queries to result
            circuit_caching (bool, optional): Use CircuitCache when calling compile_and_run_circuits
            cache_file(str, optional): filename into which to store the cache as a pickle file
            skip_qobj_deepcopy (bool, optional): Reuses the same Qobj object
                                                 over and over to avoid deepcopying
            skip_qobj_validation (bool, optional): Bypass Qobj validation to
                                                    decrease submission time
            measurement_error_mitigation_cls (Callable, optional): the approach to mitigate
                                                                    measurement
                                                                    error, CompleteMeasFitter or
                                                                    TensoredMeasFitter
            cals_matrix_refresh_period (int, optional): how long to refresh the calibration
                                                        matrix in measurement mitigation,
                                                        unit in minutes
            measurement_error_mitigation_shots (int, optional): the shot number for building
                                                                calibration matrix,
                                                                if None, use the shot number
                                                                in quantum instance
            job_callback (Callable, optional): callback used in querying info of
                                               the submitted job, and
                                               providing the following arguments: job_id,
                                               job_status, queue_position, job

        Raises:
            AquaError: the shots exceeds the maximum number of shots
            AquaError: set noise model but the backend does not support that
            AquaError: set backend_options but the backend does not support that
        """
        self._backend = backend

        # setup run config
        if shots is not None:
            if self.is_statevector and shots != 1:
                logger.info("statevector backend only works with shot=1, change "
                            "shots from %s to 1.", shots)
                shots = 1
            max_shots = self._backend.configuration().max_shots
            if max_shots is not None and shots > max_shots:
                raise AquaError('the maximum shots supported by the selected backend is {} '
                                'but you specifiy {}'.format(max_shots, shots))

        run_config = RunConfig(shots=shots, max_credits=max_credits)
        if seed_simulator:
            run_config.seed_simulator = seed_simulator

        self._run_config = run_config

        # setup backend config
        basis_gates = basis_gates or backend.configuration().basis_gates
        coupling_map = coupling_map or getattr(backend.configuration(), 'coupling_map', None)
        self._backend_config = {
            'basis_gates': basis_gates,
            'coupling_map': coupling_map
        }

        # setup compile config
        self._compile_config = {
            'pass_manager': pass_manager,
            'initial_layout': initial_layout,
            'seed_transpiler': seed_transpiler,
            'optimization_level': optimization_level
        }

        # setup job config
        self._qjob_config = {'timeout': timeout} if self.is_local \
            else {'timeout': timeout, 'wait': wait}

        # setup noise config
        self._noise_config = {}
        if noise_model is not None:
            if is_aer_qasm(self._backend):
                self._noise_config = {'noise_model': noise_model}
            else:
                raise AquaError("The noise model is not supported on the selected backend {} ({}) "
                                "only certain backends, such as Aer qasm "
                                "support noise.".format(self.backend_name,
                                                        self._backend.provider()))

        # setup backend options for run
        self._backend_options = {}
        if backend_options is not None:
            if support_backend_options(self._backend):
                self._backend_options = {'backend_options': backend_options}
            else:
                raise AquaError("backend_options can not used with the backends in IBMQ provider.")

        # setup measurement error mitigation
        self._meas_error_mitigation_cls = None
        if self.is_statevector:
            if measurement_error_mitigation_cls is not None:
                raise AquaError("Measurement error mitigation does not work "
                                "with the statevector simulation.")
        else:
            self._meas_error_mitigation_cls = measurement_error_mitigation_cls
        self._meas_error_mitigation_fitters = {}
        # TODO: support different fitting method in error mitigation?
        self._meas_error_mitigation_method = 'least_squares'
        self._cals_matrix_refresh_period = cals_matrix_refresh_period
        self._meas_error_mitigation_shots = measurement_error_mitigation_shots

        if self._meas_error_mitigation_cls is not None:
            logger.info("The measurement error mitigation is enable. "
                        "It will automatically submit an additional job to help "
                        "calibrate the result of other jobs. "
                        "The current approach will submit a job with 2^N circuits "
                        "to build the calibration matrix, "
                        "where N is the number of measured qubits. "
                        "Furthermore, Aqua will re-use the calibration matrix for %s minutes "
                        "and re-build it after that.", self._cals_matrix_refresh_period)

        # setup others
        if os.environ.get('QISKIT_AQUA_CIRCUIT_CACHE', False) or circuit_caching:
            warnings.warn("circuit_caching will be removed at Qiskit Aqua 0.7+, "
                          "this setting will be ignored. "
                          "On the other hand, Qiskit Aqua does support "
                          "parameterized circuits for adaptive algorithms (e.g. VQE and VQC) "
                          "to avoid for compiling the circuit with the same topology "
                          "multiple times",
                          UserWarning)
        if skip_qobj_deepcopy:
            warnings.warn("skip_qobj_deepcopy was used along with circuit_caching, and since "
                          "circuit_cache will be removed at Qiskit Aqua 0.7+, "
                          "this setting will be ignored, too.",
                          UserWarning)
        if cache_file:
            warnings.warn("cache_file was used along with circuit_caching, and since "
                          "circuit_cache will be removed at Qiskit Aqua 0.7+, "
                          "this setting will be ignored, too.",
                          UserWarning)

        if is_ibmq_provider(self._backend):
            if skip_qobj_validation:
                logger.warning("The skip Qobj validation does not work "
                               "for IBMQ provider. Disable it.")
                skip_qobj_validation = False
        self._skip_qobj_validation = skip_qobj_validation
        self._circuit_summary = False
        self._job_callback = job_callback
        logger.info(self)

    def __str__(self):
        """Overload string.

        Returns:
            str: the info of the object.
        """
        # pylint: disable=import-outside-toplevel
        from qiskit import __version__ as terra_version

        info = "\nQiskit Terra version: {}\n".format(terra_version)
        info += "Backend: '{} ({})', with following setting:\n{}\n{}\n{}\n{}\n{}\n{}".format(
            self.backend_name, self._backend.provider(), self._backend_config, self._compile_config,
            self._run_config, self._qjob_config, self._backend_options, self._noise_config)
        info += "\nMeasurement mitigation: {}".format(self._meas_error_mitigation_cls)

        return info

    def transpile(self, circuits):
        """
        A wrapper to transpile circuits to allow algorithm access the transpiled circuits.
        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to transpile
        Returns:
            list[QuantumCircuit]: the transpiled circuits, it is always a list even though
                                  the length is one.
        """
        transpiled_circuits = compiler.transpile(circuits, self._backend, **self._backend_config,
                                                 **self._compile_config)
        if not isinstance(transpiled_circuits, list):
            transpiled_circuits = [transpiled_circuits]

        if logger.isEnabledFor(logging.DEBUG) and self._circuit_summary:
            logger.debug("==== Before transpiler ====")
            logger.debug(summarize_circuits(circuits))
            if transpiled_circuits is not None:
                logger.debug("====  After transpiler ====")
                logger.debug(summarize_circuits(transpiled_circuits))

        return transpiled_circuits

    def execute(self, circuits, had_transpiled=False):
        """
        A wrapper to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
            had_transpiled (bool, optional): whether or not circuits had been transpiled

        Returns:
            Result: Result object

        TODO: Maybe we can combine the circuits for the main ones and calibration circuits before
              assembling to the qobj.
        """
        # pylint: disable=import-outside-toplevel
        from .utils.run_circuits import run_qobj

        from .utils.measurement_error_mitigation import (get_measured_qubits_from_qobj,
                                                         build_measurement_error_mitigation_qobj)
        # maybe compile
        if not had_transpiled:
            circuits = self.transpile(circuits)

        # assemble
        qobj = compiler.assemble(circuits, **self._run_config.to_dict())

        if self._meas_error_mitigation_cls is not None:
            qubit_index = get_measured_qubits_from_qobj(qobj)
            qubit_index_str = '_'.join([str(x) for x in qubit_index]) + \
                "_{}".format(self._meas_error_mitigation_shots or self._run_config.shots)
            meas_error_mitigation_fitter, timestamp = \
                self._meas_error_mitigation_fitters.get(qubit_index_str, (None, 0))

            if meas_error_mitigation_fitter is None:
                # check the asked qubit_index are the subset of build matrices
                for key, _ in self._meas_error_mitigation_fitters.items():
                    stored_qubit_index = [int(x) for x in key.split("_")[:-1]]
                    stored_shots = int(key.split("_")[-1])
                    if len(qubit_index) < len(stored_qubit_index):
                        tmp = list(set(qubit_index + stored_qubit_index))
                        if sorted(tmp) == sorted(stored_qubit_index) and \
                                self._run_config.shots == stored_shots:
                            # the qubit used in current job is the subset and shots are the same
                            meas_error_mitigation_fitter, timestamp = \
                                self._meas_error_mitigation_fitters.get(key, (None, 0))
                            meas_error_mitigation_fitter = \
                                meas_error_mitigation_fitter.subset_fitter(
                                    qubit_sublist=qubit_index)
                            logger.info("The qubits used in the current job is the subset of "
                                        "previous jobs, "
                                        "reusing the calibration matrix if it is not out-of-date.")

            build_cals_matrix = self.maybe_refresh_cals_matrix(timestamp) or \
                meas_error_mitigation_fitter is None

            if build_cals_matrix:
                logger.info("Updating qobj with the circuits for measurement error mitigation.")
                use_different_shots = not (
                    self._meas_error_mitigation_shots is None or
                    self._meas_error_mitigation_shots == self._run_config.shots)
                temp_run_config = copy.deepcopy(self._run_config)
                if use_different_shots:
                    temp_run_config.shots = self._meas_error_mitigation_shots

                cals_qobj, state_labels, circuit_labels = \
                    build_measurement_error_mitigation_qobj(qubit_index,
                                                            self._meas_error_mitigation_cls,
                                                            self._backend,
                                                            self._backend_config,
                                                            self._compile_config,
                                                            temp_run_config)
                if use_different_shots:
                    cals_result = run_qobj(cals_qobj, self._backend, self._qjob_config,
                                           self._backend_options,
                                           self._noise_config,
                                           self._skip_qobj_validation, self._job_callback)
                    result = run_qobj(qobj, self._backend, self._qjob_config,
                                      self._backend_options, self._noise_config,
                                      self._skip_qobj_validation, self._job_callback)
                else:
                    # insert the calibration circuit into main qobj if the shots are the same
                    qobj.experiments[0:0] = cals_qobj.experiments
                    result = run_qobj(qobj, self._backend, self._qjob_config,
                                      self._backend_options, self._noise_config,
                                      self._skip_qobj_validation, self._job_callback)
                    cals_result = result

                logger.info("Building calibration matrix for measurement error mitigation.")
                meas_error_mitigation_fitter = \
                    self._meas_error_mitigation_cls(cals_result,
                                                    state_labels,
                                                    qubit_list=qubit_index,
                                                    circlabel=circuit_labels)
                self._meas_error_mitigation_fitters[qubit_index_str] = \
                    (meas_error_mitigation_fitter, time.time())
            else:
                result = run_qobj(qobj, self._backend, self._qjob_config,
                                  self._backend_options, self._noise_config,
                                  self._skip_qobj_validation, self._job_callback)

            if meas_error_mitigation_fitter is not None:
                logger.info("Performing measurement error mitigation.")
                result = \
                    meas_error_mitigation_fitter.filter.apply(result,
                                                              self._meas_error_mitigation_method)
        else:
            result = run_qobj(qobj, self._backend, self._qjob_config,
                              self._backend_options, self._noise_config,
                              self._skip_qobj_validation, self._job_callback)

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
                if not support_backend_options(self._backend):
                    raise AquaError("backend_options can not be used with this backends "
                                    "{} ({}).".format(self.backend_name, self._backend.provider()))

                if k in QuantumInstance.BACKEND_OPTIONS_QASM_ONLY and self.is_statevector:
                    raise AquaError("'{}' is only applicable for qasm simulator but "
                                    "statevector simulator is used as the backend.")

                if 'backend_options' not in self._backend_options:
                    self._backend_options['backend_options'] = {}
                self._backend_options['backend_options'][k] = v
            elif k in QuantumInstance.NOISE_CONFIG:
                if not is_aer_qasm(self._backend):
                    raise AquaError(
                        "The noise model is not supported on the selected backend {} ({}) "
                        "only certain backends, such as Aer qasm support "
                        "noise.".format(self.backend_name, self._backend.provider()))

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
    def circuit_summary(self):
        """Getter of circuit summary."""
        return self._circuit_summary

    @circuit_summary.setter
    def circuit_summary(self, new_value):
        """ sets circuit summary """
        self._circuit_summary = new_value

    @property
    def measurement_error_mitigation_cls(self):  # pylint: disable=invalid-name
        """ returns measurement error mitigation cls """
        return self._meas_error_mitigation_cls

    @measurement_error_mitigation_cls.setter
    def measurement_error_mitigation_cls(self, new_value):  # pylint: disable=invalid-name
        """ sets measurement error mitigation cls """
        self._meas_error_mitigation_cls = new_value

    @property
    def cals_matrix_refresh_period(self):
        """ returns matrix refresh period """
        return self._cals_matrix_refresh_period

    @cals_matrix_refresh_period.setter
    def cals_matrix_refresh_period(self, new_value):
        """ sets matrix refresh period """
        self._cals_matrix_refresh_period = new_value

    @property
    def measurement_error_mitigation_shots(self):  # pylint: disable=invalid-name
        """ returns measurement error mitigation shots """
        return self._meas_error_mitigation_shots

    @measurement_error_mitigation_shots.setter
    def measurement_error_mitigation_shots(self, new_value):  # pylint: disable=invalid-name
        """ sets measurement error mitigation shots """
        self._meas_error_mitigation_shots = new_value

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
    def skip_qobj_validation(self):
        """ checks if skip qobj validation """
        return self._skip_qobj_validation

    @skip_qobj_validation.setter
    def skip_qobj_validation(self, new_value):
        """ sets skip qobj validation flag """
        self._skip_qobj_validation = new_value

    def maybe_refresh_cals_matrix(self, timestamp=None):
        """
        Calculate the time difference from the query of last time.

        Returns:
            bool: whether or not refresh the cals_matrix
        """
        timestamp = timestamp or 0
        ret = False
        curr_timestamp = time.time()
        difference = int(curr_timestamp - timestamp) / 60.0
        if difference > self._cals_matrix_refresh_period:
            ret = True

        return ret

    def cals_matrix(self, qubit_index=None):
        """
        Get the stored calibration matrices and its timestamp.

        Args:
            qubit_index (list[int]): the qubit index of corresponding calibration matrix.
            If None, return all stored calibration matrices.

        Returns:
            tuple(np.ndarray, int): the calibration matrix and the creation timestamp if qubit_index
                                    is not None otherwise, return all matrices and their timestamp
                                    in a dictionary.
        """
        ret = None
        shots = self._meas_error_mitigation_shots or self._run_config.shots
        if qubit_index:
            qubit_index_str = '_'.join([str(x) for x in qubit_index]) + "_{}".format(shots)
            fitter, timestamp = self._meas_error_mitigation_fitters.get(qubit_index_str, None)
            if fitter is not None:
                ret = (fitter.cal_matrix, timestamp)
        else:
            ret = {k: (v.cal_matrix, t) for k, (v, t)
                   in self._meas_error_mitigation_fitters.items()}
        return ret
