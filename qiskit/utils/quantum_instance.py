# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Quantum Instance module """

from typing import Optional, List, Union, Dict, Callable, Tuple
import copy
import logging
import time
import numpy as np

from qiskit.qobj import Qobj
from qiskit.utils import circuit_utils
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from .backend_utils import (
    is_ibmq_provider,
    is_statevector_backend,
    is_simulator_backend,
    is_local_backend,
    is_aer_qasm,
    is_basicaer_provider,
    support_backend_options,
)

logger = logging.getLogger(__name__)


class QuantumInstance:
    """Quantum Backend including execution setting."""

    _BACKEND_CONFIG = ["basis_gates", "coupling_map"]
    _COMPILE_CONFIG = ["initial_layout", "seed_transpiler", "optimization_level"]
    _RUN_CONFIG = ["shots", "max_credits", "memory", "seed_simulator"]
    _QJOB_CONFIG = ["timeout", "wait"]
    _NOISE_CONFIG = ["noise_model"]

    # https://github.com/Qiskit/qiskit-aer/blob/master/qiskit/providers/aer/backends/qasm_simulator.py
    _BACKEND_OPTIONS_QASM_ONLY = ["statevector_sample_measure_opt", "max_parallel_shots"]
    _BACKEND_OPTIONS = [
        "initial_statevector",
        "chop_threshold",
        "max_parallel_threads",
        "max_parallel_experiments",
        "statevector_parallel_threshold",
        "statevector_hpc_gate_opt",
    ] + _BACKEND_OPTIONS_QASM_ONLY

    def __init__(
        self,
        backend,
        # run config
        shots: Optional[int] = None,
        seed_simulator: Optional[int] = None,
        max_credits: int = 10,
        # backend properties
        basis_gates: Optional[List[str]] = None,
        coupling_map=None,
        # transpile
        initial_layout=None,
        pass_manager=None,
        seed_transpiler: Optional[int] = None,
        optimization_level: Optional[int] = None,
        # simulation
        backend_options: Optional[Dict] = None,
        noise_model=None,
        # job
        timeout: Optional[float] = None,
        wait: float = 5.0,
        # others
        skip_qobj_validation: bool = True,
        measurement_error_mitigation_cls: Optional[Callable] = None,
        cals_matrix_refresh_period: int = 30,
        measurement_error_mitigation_shots: Optional[int] = None,
        job_callback: Optional[Callable] = None,
        mit_pattern: Optional[List[List[int]]] = None,
    ) -> None:
        """
        Quantum Instance holds a Qiskit Terra backend as well as configuration for circuit
        transpilation and execution. When provided to an Aqua algorithm the algorithm will
        execute the circuits it needs to run using the instance.

        Args:
            backend (Union['Backend', 'BaseBackend']): Instance of selected backend
            shots: Number of repetitions of each circuit, for sampling. If None, the shots are
                extracted from the backend. If the backend has none set, the default is 1024.
            seed_simulator: Random seed for simulators
            max_credits: Maximum credits to use
            basis_gates: List of basis gate names supported by the
                                               target. Defaults to basis gates of the backend.
            coupling_map (Optional[Union['CouplingMap', List[List]]]):
                        Coupling map (perhaps custom) to target in mapping
            initial_layout (Optional[Union['Layout', Dict, List]]):
                        Initial layout of qubits in mapping
            pass_manager (Optional['PassManager']):
                        Pass manager to handle how to compile the circuits
            seed_transpiler: The random seed for circuit mapper
            optimization_level: How much optimization to perform on the circuits.
                Higher levels generate more optimized circuits, at the expense of longer
                transpilation time.
            backend_options: All running options for backend, please refer
                to the provider of the backend for information as to what options it supports.
            noise_model (Optional['NoiseModel']): noise model for simulator
            timeout: Seconds to wait for job. If None, wait indefinitely.
            wait: Seconds between queries for job result
            skip_qobj_validation: Bypass Qobj validation to decrease circuit
                processing time during submission to backend.
            measurement_error_mitigation_cls: The approach to mitigate
                measurement errors. Qiskit Ignis provides fitter classes for this functionality
                and CompleteMeasFitter or TensoredMeasFitter
                from qiskit.ignis.mitigation.measurement module can be used here.
                TensoredMeasFitter doesn't support subset fitter.
            cals_matrix_refresh_period: How often to refresh the calibration
                matrix in measurement mitigation. in minutes
            measurement_error_mitigation_shots: The number of shots number for
                building calibration matrix. If None, the main `shots` parameter value is used.
            job_callback: Optional user supplied callback which can be used
                to monitor job progress as jobs are submitted for processing by an Aqua algorithm.
                The callback is provided the following arguments: `job_id, job_status,
                queue_position, job`
            mit_pattern: Qubits on which to perform the TensoredMeasFitter
                measurement correction, divided to groups according to tensors.
                If `None` and `qr` is given then assumed to be performed over the entire
                `qr` as one group (default `None`).

        Raises:
            QiskitError: the shots exceeds the maximum number of shots
            QiskitError: set noise model but the backend does not support that
            QiskitError: set backend_options but the backend does not support that
        """
        self._backend = backend
        self._pass_manager = pass_manager

        # if the shots are none, try to get them from the backend
        if shots is None:
            from qiskit.providers.basebackend import BaseBackend  # pylint: disable=cyclic-import
            from qiskit.providers.backend import BackendV1  # pylint: disable=cyclic-import

            if isinstance(backend, (BaseBackend, BackendV1)):
                if hasattr(backend, "options"):  # should always be true for V1
                    backend_shots = backend.options.get("shots", 1024)
                    if shots != backend_shots:
                        logger.info(
                            "Overwriting the number of shots in the quantum instance with "
                            "the settings from the backend."
                        )
                    shots = backend_shots

        # safeguard if shots are still not set
        if shots is None:
            shots = 1024

        # pylint: disable=cyclic-import
        from qiskit.assembler.run_config import RunConfig

        run_config = RunConfig(shots=shots, max_credits=max_credits)
        if seed_simulator is not None:
            run_config.seed_simulator = seed_simulator

        self._run_config = run_config

        # setup backend config
        basis_gates = basis_gates or backend.configuration().basis_gates
        coupling_map = coupling_map or getattr(backend.configuration(), "coupling_map", None)
        self._backend_config = {"basis_gates": basis_gates, "coupling_map": coupling_map}

        # setup compile config
        self._compile_config = {
            "initial_layout": initial_layout,
            "seed_transpiler": seed_transpiler,
            "optimization_level": optimization_level,
        }

        # setup job config
        self._qjob_config = (
            {"timeout": timeout} if self.is_local else {"timeout": timeout, "wait": wait}
        )

        # setup noise config
        self._noise_config = {}
        if noise_model is not None:
            if is_simulator_backend(self._backend) and not is_basicaer_provider(self._backend):
                self._noise_config = {"noise_model": noise_model}
            else:
                raise QiskitError(
                    "The noise model is not supported "
                    "on the selected backend {} ({}) "
                    "only certain backends, such as Aer qasm simulator "
                    "support noise.".format(self.backend_name, self._backend.provider())
                )

        # setup backend options for run
        self._backend_options = {}
        if backend_options is not None:
            if support_backend_options(self._backend):
                self._backend_options = {"backend_options": backend_options}
            else:
                raise QiskitError(
                    "backend_options can not used with the backends in " "IBMQ provider."
                )

        # setup measurement error mitigation
        self._meas_error_mitigation_cls = None
        if self.is_statevector:
            if measurement_error_mitigation_cls is not None:
                raise QiskitError(
                    "Measurement error mitigation does not work " "with the statevector simulation."
                )
        else:
            self._meas_error_mitigation_cls = measurement_error_mitigation_cls
        self._meas_error_mitigation_fitters: Dict[str, Tuple[np.ndarray, float]] = {}
        # TODO: support different fitting method in error mitigation?
        self._meas_error_mitigation_method = "least_squares"
        self._cals_matrix_refresh_period = cals_matrix_refresh_period
        self._meas_error_mitigation_shots = measurement_error_mitigation_shots
        self._mit_pattern = mit_pattern

        if self._meas_error_mitigation_cls is not None:
            logger.info(
                "The measurement error mitigation is enabled. "
                "It will automatically submit an additional job to help "
                "calibrate the result of other jobs. "
                "The current approach will submit a job with 2^N circuits "
                "to build the calibration matrix, "
                "where N is the number of measured qubits. "
                "Furthermore, Aqua will re-use the calibration matrix for %s minutes "
                "and re-build it after that.",
                self._cals_matrix_refresh_period,
            )

        # setup others
        if is_ibmq_provider(self._backend):
            if skip_qobj_validation:
                logger.info(
                    "skip_qobj_validation was set True but this setting is not "
                    "supported by IBMQ provider and has been ignored."
                )
                skip_qobj_validation = False
        self._skip_qobj_validation = skip_qobj_validation
        self._circuit_summary = False
        self._job_callback = job_callback
        self._time_taken = 0.0
        logger.info(self)

    def __str__(self) -> str:
        """Overload string.

        Returns:
            str: the info of the object.
        """
        from qiskit import __version__ as terra_version

        info = f"\nQiskit Terra version: {terra_version}\n"
        info += "Backend: '{} ({})', with following setting:\n{}\n{}\n{}\n{}\n{}\n{}".format(
            self.backend_name,
            self._backend.provider(),
            self._backend_config,
            self._compile_config,
            self._run_config,
            self._qjob_config,
            self._backend_options,
            self._noise_config,
        )
        info += f"\nMeasurement mitigation: {self._meas_error_mitigation_cls}"

        return info

    def transpile(self, circuits):
        """
        A wrapper to transpile circuits to allow algorithm access the transpiled circuits.
        Args:
            circuits (Union['QuantumCircuit', List['QuantumCircuit']]): circuits to transpile
        Returns:
            List['QuantumCircuit']: The transpiled circuits, it is always a list even though
                                    the length is one.
        """
        # pylint: disable=cyclic-import
        from qiskit import compiler

        if self._pass_manager is not None:
            transpiled_circuits = self._pass_manager.run(circuits)
        else:
            transpiled_circuits = compiler.transpile(
                circuits, self._backend, **self._backend_config, **self._compile_config
            )
        if not isinstance(transpiled_circuits, list):
            transpiled_circuits = [transpiled_circuits]

        if logger.isEnabledFor(logging.DEBUG) and self._circuit_summary:
            logger.debug("==== Before transpiler ====")
            logger.debug(circuit_utils.summarize_circuits(circuits))
            if transpiled_circuits is not None:
                logger.debug("====  After transpiler ====")
                logger.debug(circuit_utils.summarize_circuits(transpiled_circuits))

        return transpiled_circuits

    def assemble(self, circuits) -> Qobj:
        """assemble circuits"""
        # pylint: disable=cyclic-import
        from qiskit import compiler

        return compiler.assemble(circuits, **self._run_config.to_dict())

    def execute(self, circuits, had_transpiled: bool = False):
        """
        A wrapper to interface with quantum backend.

        Args:
            circuits (Union['QuantumCircuit', List['QuantumCircuit']]):
                        circuits to execute
            had_transpiled: whether or not circuits had been transpiled

        Raises:
            QiskitError: Invalid error mitigation fitter class
            QiskitError: TensoredMeasFitter class doesn't support subset fitter
            MissingOptionalLibraryError: Ignis not installed

        Returns:
            Result: result object

        TODO: Maybe we can combine the circuits for the main ones and calibration circuits before
              assembling to the qobj.
        """
        from qiskit.utils.run_circuits import run_qobj, run_circuits

        from qiskit.utils.measurement_error_mitigation import (
            get_measured_qubits_from_qobj,
            get_measured_qubits,
            build_measurement_error_mitigation_circuits,
            build_measurement_error_mitigation_qobj,
        )

        # maybe compile
        if not had_transpiled:
            circuits = self.transpile(circuits)

        from qiskit.providers import BackendV1

        circuit_job = isinstance(self._backend, BackendV1)
        if self.is_statevector and self._backend.name() == "aer_simulator_statevector":
            try:
                from qiskit.providers.aer.library import SaveStatevector

                def _find_save_state(data):
                    for instr, _, _ in reversed(data):
                        if isinstance(instr, SaveStatevector):
                            return True
                    return False

                if isinstance(circuits, list):
                    for circuit in circuits:
                        if not _find_save_state(circuit.data):
                            circuit.save_statevector()
                else:
                    if not _find_save_state(circuits.data):
                        circuits.save_statevector()
            except ImportError:
                pass

        # assemble
        if not circuit_job:
            qobj = self.assemble(circuits)

        if self._meas_error_mitigation_cls is not None:
            qubit_index, qubit_mappings = (
                get_measured_qubits(circuits)
                if circuit_job
                else get_measured_qubits_from_qobj(qobj)
            )
            mit_pattern = self._mit_pattern
            if mit_pattern is None:
                mit_pattern = [[i] for i in range(len(qubit_index))]
            qubit_index_str = "_".join([str(x) for x in qubit_index]) + "_{}".format(
                self._meas_error_mitigation_shots or self._run_config.shots
            )
            meas_error_mitigation_fitter, timestamp = self._meas_error_mitigation_fitters.get(
                qubit_index_str, (None, 0.0)
            )

            if meas_error_mitigation_fitter is None:
                # check the asked qubit_index are the subset of build matrices
                for key, _ in self._meas_error_mitigation_fitters.items():
                    stored_qubit_index = [int(x) for x in key.split("_")[:-1]]
                    stored_shots = int(key.split("_")[-1])
                    if len(qubit_index) < len(stored_qubit_index):
                        tmp = list(set(qubit_index + stored_qubit_index))
                        if (
                            sorted(tmp) == sorted(stored_qubit_index)
                            and self._run_config.shots == stored_shots
                        ):
                            # the qubit used in current job is the subset and shots are the same
                            (
                                meas_error_mitigation_fitter,
                                timestamp,
                            ) = self._meas_error_mitigation_fitters.get(key, (None, 0.0))
                            meas_error_mitigation_fitter = (
                                meas_error_mitigation_fitter.subset_fitter(
                                    qubit_sublist=qubit_index
                                )
                            )
                            logger.info(
                                "The qubits used in the current job is the subset of "
                                "previous jobs, "
                                "reusing the calibration matrix if it is not out-of-date."
                            )

            build_cals_matrix = (
                self.maybe_refresh_cals_matrix(timestamp) or meas_error_mitigation_fitter is None
            )

            if build_cals_matrix:
                if circuit_job:
                    logger.info("Updating to also run measurement error mitigation.")
                    use_different_shots = not (
                        self._meas_error_mitigation_shots is None
                        or self._meas_error_mitigation_shots == self._run_config.shots
                    )
                    temp_run_config = copy.deepcopy(self._run_config)
                    if use_different_shots:
                        temp_run_config.shots = self._meas_error_mitigation_shots
                    (
                        cal_circuits,
                        state_labels,
                        circuit_labels,
                    ) = build_measurement_error_mitigation_circuits(
                        qubit_index,
                        self._meas_error_mitigation_cls,
                        self._backend,
                        self._backend_config,
                        self._compile_config,
                        mit_pattern=mit_pattern,
                    )
                    if use_different_shots:
                        cals_result = run_circuits(
                            cal_circuits,
                            self._backend,
                            qjob_config=self._qjob_config,
                            backend_options=self._backend_options,
                            noise_config=self._noise_config,
                            run_config=self._run_config.to_dict(),
                            job_callback=self._job_callback,
                        )
                        self._time_taken += cals_result.time_taken
                        result = run_circuits(
                            circuits,
                            self._backend,
                            qjob_config=self.qjob_config,
                            backend_options=self.backend_options,
                            noise_config=self._noise_config,
                            run_config=self.run_config.to_dict(),
                            job_callback=self._job_callback,
                        )
                        self._time_taken += result.time_taken
                    else:
                        circuits[0:0] = cal_circuits
                        result = run_circuits(
                            circuits,
                            self._backend,
                            qjob_config=self.qjob_config,
                            backend_options=self.backend_options,
                            noise_config=self._noise_config,
                            run_config=self.run_config.to_dict(),
                            job_callback=self._job_callback,
                        )
                        self._time_taken += result.time_taken
                        cals_result = result

                else:
                    logger.info("Updating qobj with the circuits for measurement error mitigation.")
                    use_different_shots = not (
                        self._meas_error_mitigation_shots is None
                        or self._meas_error_mitigation_shots == self._run_config.shots
                    )
                    temp_run_config = copy.deepcopy(self._run_config)
                    if use_different_shots:
                        temp_run_config.shots = self._meas_error_mitigation_shots

                    (
                        cals_qobj,
                        state_labels,
                        circuit_labels,
                    ) = build_measurement_error_mitigation_qobj(
                        qubit_index,
                        self._meas_error_mitigation_cls,
                        self._backend,
                        self._backend_config,
                        self._compile_config,
                        temp_run_config,
                        mit_pattern=mit_pattern,
                    )
                    if use_different_shots or is_aer_qasm(self._backend):
                        cals_result = run_qobj(
                            cals_qobj,
                            self._backend,
                            self._qjob_config,
                            self._backend_options,
                            self._noise_config,
                            self._skip_qobj_validation,
                            self._job_callback,
                        )
                        self._time_taken += cals_result.time_taken
                        result = run_qobj(
                            qobj,
                            self._backend,
                            self._qjob_config,
                            self._backend_options,
                            self._noise_config,
                            self._skip_qobj_validation,
                            self._job_callback,
                        )
                        self._time_taken += result.time_taken
                    else:
                        # insert the calibration circuit into main qobj if the shots are the same
                        qobj.experiments[0:0] = cals_qobj.experiments
                        result = run_qobj(
                            qobj,
                            self._backend,
                            self._qjob_config,
                            self._backend_options,
                            self._noise_config,
                            self._skip_qobj_validation,
                            self._job_callback,
                        )
                        self._time_taken += result.time_taken
                        cals_result = result

                logger.info("Building calibration matrix for measurement error mitigation.")
                try:
                    from qiskit.ignis.mitigation.measurement import (
                        CompleteMeasFitter,
                        TensoredMeasFitter,
                    )
                except ImportError as ex:
                    raise MissingOptionalLibraryError(
                        libname="qiskit-ignis",
                        name="execute",
                        pip_install="pip install qiskit-ignis",
                    ) from ex
                if self._meas_error_mitigation_cls == CompleteMeasFitter:
                    meas_error_mitigation_fitter = self._meas_error_mitigation_cls(
                        cals_result, state_labels, qubit_list=qubit_index, circlabel=circuit_labels
                    )
                elif self._meas_error_mitigation_cls == TensoredMeasFitter:
                    meas_error_mitigation_fitter = self._meas_error_mitigation_cls(
                        cals_result, mit_pattern=state_labels, circlabel=circuit_labels
                    )
                else:
                    raise QiskitError(f"Unknown fitter {self._meas_error_mitigation_cls}")

                self._meas_error_mitigation_fitters[qubit_index_str] = (
                    meas_error_mitigation_fitter,
                    time.time(),
                )
            else:
                result = (
                    run_circuits(
                        circuits,
                        self._backend,
                        qjob_config=self.qjob_config,
                        backend_options=self.backend_options,
                        noise_config=self._noise_config,
                        run_config=self._run_config.to_dict(),
                        job_callback=self._job_callback,
                    )
                    if circuit_job
                    else run_qobj(
                        qobj,
                        self._backend,
                        self._qjob_config,
                        self._backend_options,
                        self._noise_config,
                        self._skip_qobj_validation,
                        self._job_callback,
                    )
                )
                self._time_taken += result.time_taken

            if meas_error_mitigation_fitter is not None:
                logger.info("Performing measurement error mitigation.")
                if (
                    hasattr(self._run_config, "parameterizations")
                    and len(self._run_config.parameterizations) > 0
                    and len(self._run_config.parameterizations[0]) > 0
                    and len(self._run_config.parameterizations[0][0]) > 0
                ):
                    num_circuit_templates = len(self._run_config.parameterizations)
                    num_param_variations = len(self._run_config.parameterizations[0][0])
                    num_circuits = num_circuit_templates * num_param_variations
                else:
                    num_circuits = len(circuits)
                skip_num_circuits = len(result.results) - num_circuits
                #  remove the calibration counts from result object to assure the length of
                #  ExperimentalResult is equal length to input circuits
                result.results = result.results[skip_num_circuits:]
                tmp_result = copy.deepcopy(result)
                for qubit_index_str, c_idx in qubit_mappings.items():
                    curr_qubit_index = [int(x) for x in qubit_index_str.split("_")]
                    tmp_result.results = [result.results[i] for i in c_idx]
                    if curr_qubit_index == qubit_index:
                        tmp_fitter = meas_error_mitigation_fitter
                    elif isinstance(meas_error_mitigation_fitter, CompleteMeasFitter):
                        tmp_fitter = meas_error_mitigation_fitter.subset_fitter(curr_qubit_index)
                    else:
                        raise QiskitError(
                            "{} doesn't support subset_fitter.".format(
                                meas_error_mitigation_fitter.__class__.__name__
                            )
                        )
                    tmp_result = tmp_fitter.filter.apply(
                        tmp_result, self._meas_error_mitigation_method
                    )
                    for i, n in enumerate(c_idx):
                        result.results[n] = tmp_result.results[i]

        else:
            result = (
                run_circuits(
                    circuits,
                    self._backend,
                    qjob_config=self.qjob_config,
                    backend_options=self.backend_options,
                    noise_config=self._noise_config,
                    run_config=self._run_config.to_dict(),
                    job_callback=self._job_callback,
                )
                if circuit_job
                else run_qobj(
                    qobj,
                    self._backend,
                    self._qjob_config,
                    self._backend_options,
                    self._noise_config,
                    self._skip_qobj_validation,
                    self._job_callback,
                )
            )
            self._time_taken += result.time_taken

        if self._circuit_summary:
            self._circuit_summary = False

        return result

    def set_config(self, **kwargs):
        """Set configurations for the quantum instance."""
        for k, v in kwargs.items():
            if k in QuantumInstance._RUN_CONFIG:
                setattr(self._run_config, k, v)
            elif k in QuantumInstance._QJOB_CONFIG:
                self._qjob_config[k] = v
            elif k in QuantumInstance._COMPILE_CONFIG:
                self._compile_config[k] = v
            elif k in QuantumInstance._BACKEND_CONFIG:
                self._backend_config[k] = v
            elif k in QuantumInstance._BACKEND_OPTIONS:
                if not support_backend_options(self._backend):
                    raise QiskitError(
                        "backend_options can not be used with this backend "
                        "{} ({}).".format(self.backend_name, self._backend.provider())
                    )

                if k in QuantumInstance._BACKEND_OPTIONS_QASM_ONLY and self.is_statevector:
                    raise QiskitError(
                        "'{}' is only applicable for qasm simulator but "
                        "statevector simulator is used as the backend."
                    )

                if "backend_options" not in self._backend_options:
                    self._backend_options["backend_options"] = {}
                self._backend_options["backend_options"][k] = v
            elif k in QuantumInstance._NOISE_CONFIG:
                if not is_simulator_backend(self._backend) or is_basicaer_provider(self._backend):
                    raise QiskitError(
                        "The noise model is not supported on the selected backend {} ({}) "
                        "only certain backends, such as Aer qasm support "
                        "noise.".format(self.backend_name, self._backend.provider())
                    )

                self._noise_config[k] = v

            else:
                raise ValueError(f"unknown setting for the key ({k}).")

    @property
    def time_taken(self) -> float:
        """Accumulated time taken for execution."""
        return self._time_taken

    def reset_execution_results(self) -> None:
        """Reset execution results"""
        self._time_taken = 0.0

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
        """sets circuit summary"""
        self._circuit_summary = new_value

    @property
    def measurement_error_mitigation_cls(self):  # pylint: disable=invalid-name
        """returns measurement error mitigation cls"""
        return self._meas_error_mitigation_cls

    @measurement_error_mitigation_cls.setter
    def measurement_error_mitigation_cls(self, new_value):  # pylint: disable=invalid-name
        """sets measurement error mitigation cls"""
        self._meas_error_mitigation_cls = new_value

    @property
    def cals_matrix_refresh_period(self):
        """returns matrix refresh period"""
        return self._cals_matrix_refresh_period

    @cals_matrix_refresh_period.setter
    def cals_matrix_refresh_period(self, new_value):
        """sets matrix refresh period"""
        self._cals_matrix_refresh_period = new_value

    @property
    def measurement_error_mitigation_shots(self):  # pylint: disable=invalid-name
        """returns measurement error mitigation shots"""
        return self._meas_error_mitigation_shots

    @measurement_error_mitigation_shots.setter
    def measurement_error_mitigation_shots(self, new_value):  # pylint: disable=invalid-name
        """sets measurement error mitigation shots"""
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
        """checks if skip qobj validation"""
        return self._skip_qobj_validation

    @skip_qobj_validation.setter
    def skip_qobj_validation(self, new_value):
        """sets skip qobj validation flag"""
        self._skip_qobj_validation = new_value

    def maybe_refresh_cals_matrix(self, timestamp: Optional[float] = None) -> bool:
        """
        Calculate the time difference from the query of last time.

        Args:
            timestamp: timestamp

        Returns:
            Whether or not refresh the cals_matrix
        """
        timestamp = timestamp or 0.0
        ret = False
        curr_timestamp = time.time()
        difference = int(curr_timestamp - timestamp) / 60.0
        if difference > self._cals_matrix_refresh_period:
            ret = True

        return ret

    def cals_matrix(
        self, qubit_index: Optional[List[int]] = None
    ) -> Optional[Union[Tuple[np.ndarray, float], Dict[str, Tuple[np.ndarray, float]]]]:
        """
        Get the stored calibration matrices and its timestamp.

        Args:
            qubit_index: the qubit index of corresponding calibration matrix.
                         If None, return all stored calibration matrices.

        Returns:
            The calibration matrix and the creation timestamp if qubit_index
            is not None otherwise, return all matrices and their timestamp
            in a dictionary.
        """
        shots = self._meas_error_mitigation_shots or self._run_config.shots
        if qubit_index:
            qubit_index_str = "_".join([str(x) for x in qubit_index]) + f"_{shots}"
            fitter, timestamp = self._meas_error_mitigation_fitters.get(qubit_index_str, None)
            if fitter is not None:
                return fitter.cal_matrix, timestamp
        else:
            return {
                k: (v.cal_matrix, t) for k, (v, t) in self._meas_error_mitigation_fitters.items()
            }
        return None
