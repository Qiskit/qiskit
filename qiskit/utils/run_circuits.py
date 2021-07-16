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

""" run circuits functions """

from typing import Optional, Dict, Callable, List, Union, Tuple
import sys
import logging
import time
import copy
import os
import uuid

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend, BaseBackend, JobStatus, JobError, BaseJob
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.qobj import QasmQobj
from ..exceptions import QiskitError, MissingOptionalLibraryError
from .backend_utils import (
    is_aer_provider,
    is_basicaer_provider,
    is_simulator_backend,
    is_local_backend,
    is_ibmq_provider,
)

MAX_CIRCUITS_PER_JOB = os.environ.get("QISKIT_AQUA_MAX_CIRCUITS_PER_JOB", None)
MAX_GATES_PER_JOB = os.environ.get("QISKIT_AQUA_MAX_GATES_PER_JOB", None)

logger = logging.getLogger(__name__)


def find_regs_by_name(
    circuit: QuantumCircuit, name: str, qreg: bool = True
) -> Optional[Union[QuantumRegister, ClassicalRegister]]:
    """Find the registers in the circuits.

    Args:
        circuit: the quantum circuit.
        name: name of register
        qreg: quantum or classical register

    Returns:
        if not found, return None.

    """
    found_reg = None
    regs = circuit.qregs if qreg else circuit.cregs
    for reg in regs:
        if reg.name == name:
            found_reg = reg
            break
    return found_reg


def _combine_result_objects(results: List[Result]) -> Result:
    """Temporary helper function.

    TODO:
        This function would be removed after Terra supports job with infinite circuits.
    """
    if len(results) == 1:
        return results[0]

    new_result = copy.deepcopy(results[0])

    for idx in range(1, len(results)):
        new_result.results.extend(results[idx].results)

    return new_result


def _split_qobj_to_qobjs(qobj: QasmQobj, chunk_size: int) -> List[QasmQobj]:
    qobjs = []
    num_chunks = int(np.ceil(len(qobj.experiments) / chunk_size))
    if num_chunks == 1:
        qobjs = [qobj]
    else:
        if isinstance(qobj, QasmQobj):
            qobj_template = QasmQobj(
                qobj_id=qobj.qobj_id, config=qobj.config, experiments=[], header=qobj.header
            )
            for i in range(num_chunks):
                temp_qobj = copy.deepcopy(qobj_template)
                temp_qobj.qobj_id = str(uuid.uuid4())
                temp_qobj.experiments = qobj.experiments[i * chunk_size : (i + 1) * chunk_size]
                qobjs = _maybe_split_qobj_by_gates(qobjs, temp_qobj)
        else:
            raise QiskitError("Only support QasmQobj now.")

    return qobjs


def _maybe_split_qobj_by_gates(qobjs: List[QasmQobj], qobj: QasmQobj) -> List[QasmQobj]:
    if MAX_GATES_PER_JOB is not None:
        max_gates_per_job = int(MAX_GATES_PER_JOB)
        total_num_gates = 0
        for j in range(len(qobj.experiments)):
            total_num_gates += len(qobj.experiments[j].instructions)
        # split by gates if total number of gates in a qobj exceed MAX_GATES_PER_JOB
        if total_num_gates > max_gates_per_job:
            qobj_template = QasmQobj(
                qobj_id=qobj.qobj_id, config=qobj.config, experiments=[], header=qobj.header
            )
            temp_qobj = copy.deepcopy(qobj_template)
            temp_qobj.qobj_id = str(uuid.uuid4())
            temp_qobj.experiments = []
            num_gates = 0
            for i in range(len(qobj.experiments)):
                num_gates += len(qobj.experiments[i].instructions)
                if num_gates <= max_gates_per_job:
                    temp_qobj.experiments.append(qobj.experiments[i])
                else:
                    qobjs.append(temp_qobj)
                    # Initialize for next temp_qobj
                    temp_qobj = copy.deepcopy(qobj_template)
                    temp_qobj.qobj_id = str(uuid.uuid4())
                    temp_qobj.experiments.append(qobj.experiments[i])
                    num_gates = len(qobj.experiments[i].instructions)

            qobjs.append(temp_qobj)
        else:
            qobjs.append(qobj)
    else:
        qobjs.append(qobj)

    return qobjs


def _safe_submit_qobj(
    qobj: QasmQobj,
    backend: Union[Backend, BaseBackend],
    backend_options: Dict,
    noise_config: Dict,
    skip_qobj_validation: bool,
) -> Tuple[BaseJob, str]:
    # assure get job ids
    while True:
        try:
            job = run_on_backend(
                backend,
                qobj,
                backend_options=backend_options,
                noise_config=noise_config,
                skip_qobj_validation=skip_qobj_validation,
            )
            job_id = job.job_id()
            break
        except QiskitError as ex:
            failure_warn = True
            if is_ibmq_provider(backend):
                try:
                    from qiskit.providers.ibmq import IBMQBackendJobLimitError
                except ImportError as ex1:
                    raise MissingOptionalLibraryError(
                        libname="qiskit-ibmq-provider",
                        name="_safe_submit_qobj",
                        pip_install="pip install qiskit-ibmq-provider",
                    ) from ex1
                if isinstance(ex, IBMQBackendJobLimitError):

                    oldest_running = backend.jobs(
                        limit=1, descending=False, status=["QUEUED", "VALIDATING", "RUNNING"]
                    )
                    if oldest_running:
                        oldest_running = oldest_running[0]
                        logger.warning(
                            "Job limit reached, waiting for job %s to finish "
                            "before submitting the next one.",
                            oldest_running.job_id(),
                        )
                        failure_warn = False  # Don't issue a second warning.
                        try:
                            oldest_running.wait_for_final_state(timeout=300)
                        except Exception:  # pylint: disable=broad-except
                            # If the wait somehow fails or times out, we'll just re-try
                            # the job submit and see if it works now.
                            pass
            if failure_warn:
                logger.warning(
                    "FAILURE: Can not get job id, Resubmit the qobj to get job id. "
                    "Terra job error: %s ",
                    ex,
                )
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "FAILURE: Can not get job id, Resubmit the qobj to get job id." "Error: %s ", ex
            )

    return job, job_id


def _safe_get_job_status(job: BaseJob, job_id: str) -> JobStatus:

    while True:
        try:
            job_status = job.status()
            break
        except JobError as ex:
            logger.warning(
                "FAILURE: job id: %s, " "status: 'FAIL_TO_GET_STATUS' " "Terra job error: %s",
                job_id,
                ex,
            )
            time.sleep(5)
        except Exception as ex:
            raise QiskitError(
                "FAILURE: job id: {}, "
                "status: 'FAIL_TO_GET_STATUS' "
                "Unknown error: ({})".format(job_id, ex)
            ) from ex
    return job_status


def run_qobj(
    qobj: QasmQobj,
    backend: Union[Backend, BaseBackend],
    qjob_config: Optional[Dict] = None,
    backend_options: Optional[Dict] = None,
    noise_config: Optional[Dict] = None,
    skip_qobj_validation: bool = False,
    job_callback: Optional[Callable] = None,
) -> Result:
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The auto-recovery feature is only applied for non-simulator backend.
    This wrapper will try to get the result no matter how long it takes.

    Args:
        qobj: qobj to execute
        backend: backend instance
        qjob_config: configuration for quantum job object
        backend_options: configuration for simulator
        noise_config: configuration for noise model
        skip_qobj_validation: Bypass Qobj validation to decrease submission time,
                                               only works for Aer and BasicAer providers
        job_callback: callback used in querying info of the submitted job, and
                                           providing the following arguments:
                                            job_id, job_status, queue_position, job

    Returns:
        Result object

    Raises:
        ValueError: invalid backend
        QiskitError: Any error except for JobError raised by Qiskit Terra
    """
    qjob_config = qjob_config or {}
    backend_options = backend_options or {}
    noise_config = noise_config or {}

    if backend is None or not isinstance(backend, (Backend, BaseBackend)):
        raise ValueError("Backend is missing or not an instance of BaseBackend")

    with_autorecover = not is_simulator_backend(backend)

    if MAX_CIRCUITS_PER_JOB is not None:
        max_circuits_per_job = int(MAX_CIRCUITS_PER_JOB)
    else:
        if is_local_backend(backend):
            max_circuits_per_job = sys.maxsize
        else:
            max_circuits_per_job = backend.configuration().max_experiments

    # split qobj if it exceeds the payload of the backend

    qobjs = _split_qobj_to_qobjs(qobj, max_circuits_per_job)
    jobs = []
    job_ids = []
    for qob in qobjs:
        job, job_id = _safe_submit_qobj(
            qob, backend, backend_options, noise_config, skip_qobj_validation
        )
        job_ids.append(job_id)
        jobs.append(job)

    results = []
    if with_autorecover:
        logger.info("Backend status: %s", backend.status())
        logger.info("There are %s jobs are submitted.", len(jobs))
        logger.info("All job ids:\n%s", job_ids)
        for idx, _ in enumerate(jobs):
            job = jobs[idx]
            job_id = job_ids[idx]
            while True:
                logger.info("Running %s-th qobj, job id: %s", idx, job_id)
                # try to get result if possible
                while True:
                    job_status = _safe_get_job_status(job, job_id)
                    queue_position = 0
                    if job_status in JOB_FINAL_STATES:
                        # do callback again after the job is in the final states
                        if job_callback is not None:
                            job_callback(job_id, job_status, queue_position, job)
                        break
                    if job_status == JobStatus.QUEUED:
                        queue_position = job.queue_position()
                        logger.info("Job id: %s is queued at position %s", job_id, queue_position)
                    else:
                        logger.info("Job id: %s, status: %s", job_id, job_status)
                    if job_callback is not None:
                        job_callback(job_id, job_status, queue_position, job)
                    time.sleep(qjob_config["wait"])

                # get result after the status is DONE
                if job_status == JobStatus.DONE:
                    while True:
                        result = job.result(**qjob_config)
                        if result.success:
                            results.append(result)
                            logger.info("COMPLETED the %s-th qobj, job id: %s", idx, job_id)
                            break

                        logger.warning("FAILURE: Job id: %s", job_id)
                        logger.warning(
                            "Job (%s) is completed anyway, retrieve result " "from backend again.",
                            job_id,
                        )
                        job = backend.retrieve_job(job_id)
                    break
                # for other cases, resubmit the qobj until the result is available.
                # since if there is no result returned, there is no way algorithm can do any process
                # get back the qobj first to avoid for job is consumed
                qobj = job.qobj()
                if job_status == JobStatus.CANCELLED:
                    logger.warning("FAILURE: Job id: %s is cancelled. Re-submit the Qobj.", job_id)
                elif job_status == JobStatus.ERROR:
                    logger.warning(
                        "FAILURE: Job id: %s encounters the error. "
                        "Error is : %s. Re-submit the Qobj.",
                        job_id,
                        job.error_message(),
                    )
                else:
                    logging.warning(
                        "FAILURE: Job id: %s. Unknown status: %s. " "Re-submit the Qobj.",
                        job_id,
                        job_status,
                    )

                job, job_id = _safe_submit_qobj(
                    qobj, backend, backend_options, noise_config, skip_qobj_validation
                )
                jobs[idx] = job
                job_ids[idx] = job_id
    else:
        results = []
        for job in jobs:
            results.append(job.result(**qjob_config))

    result = _combine_result_objects(results) if results else None

    # If result was not successful then raise an exception with either the status msg or
    # extra information if this was an Aer partial result return
    if not result.success:
        msg = result.status
        if result.status == "PARTIAL COMPLETED":
            # Aer can return partial results which Aqua algorithms cannot process and signals
            # using partial completed status where each returned result has a success and status.
            # We use the status from the first result that was not successful
            for res in result.results:
                if not res.success:
                    msg += ", " + res.status
                    break
        raise QiskitError(f"Circuit execution failed: {msg}")

    if not hasattr(result, "time_taken"):
        setattr(result, "time_taken", 0.0)

    return result


# skip_qobj_validation = True does what backend.run
# and aerjob.submit do, but without qobj validation.
def run_on_backend(
    backend: Union[Backend, BaseBackend],
    qobj: QasmQobj,
    backend_options: Optional[Dict] = None,
    noise_config: Optional[Dict] = None,
    skip_qobj_validation: bool = False,
) -> BaseJob:
    """run on backend"""
    if skip_qobj_validation:
        if is_aer_provider(backend) or is_basicaer_provider(backend):
            if backend_options is not None:
                for option, value in backend_options.items():
                    if option == "backend_options":
                        for key, val in value.items():
                            setattr(qobj.config, key, val)
                    else:
                        setattr(qobj.config, option, value)
            if (
                is_aer_provider(backend)
                and noise_config is not None
                and "noise_model" in noise_config
            ):
                qobj.config.noise_model = noise_config["noise_model"]
            job = backend.run(qobj, validate=False)
        else:
            logger.info(
                "Can't skip qobj validation for the %s provider.",
                backend.provider().__class__.__name__,
            )
            job = backend.run(qobj, **backend_options, **noise_config)
        return job
    else:
        job = backend.run(qobj, **backend_options, **noise_config)
        return job


def run_circuits(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Union[Backend, BaseBackend],
    qjob_config: Dict,
    backend_options: Optional[Dict] = None,
    noise_config: Optional[Dict] = None,
    run_config: Optional[Dict] = None,
    job_callback: Optional[Callable] = None,
) -> Result:
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The auto-recovery feature is only applied for non-simulator backend.
    This wrapper will try to get the result no matter how long it takes.

    Args:
        circuits: circuits to execute
        backend: backend instance
        qjob_config: configuration for quantum job object
        backend_options: backend options
        noise_config: configuration for noise model
        run_config: configuration for run
        job_callback: callback used in querying info of the submitted job, and
                                           providing the following arguments:
                                            job_id, job_status, queue_position, job

    Returns:
        Result object

    Raises:
        QiskitError: Any error except for JobError raised by Qiskit Terra
    """
    backend_options = backend_options or {}
    noise_config = noise_config or {}
    run_config = run_config or {}
    with_autorecover = not is_simulator_backend(backend)

    if MAX_CIRCUITS_PER_JOB is not None:
        max_circuits_per_job = int(MAX_CIRCUITS_PER_JOB)
    else:
        if is_local_backend(backend):
            max_circuits_per_job = sys.maxsize
        else:
            max_circuits_per_job = backend.configuration().max_experiments

    if len(circuits) > max_circuits_per_job:
        jobs = []
        job_ids = []
        split_circuits = []
        count = 0
        while count < len(circuits):
            some_circuits = circuits[count : count + max_circuits_per_job]
            split_circuits.append(some_circuits)
            job, job_id = _safe_submit_circuits(
                some_circuits,
                backend,
                qjob_config=qjob_config,
                backend_options=backend_options,
                noise_config=noise_config,
                run_config=run_config,
            )
            jobs.append(job)
            job_ids.append(job_id)
            count += max_circuits_per_job
    else:
        job, job_id = _safe_submit_circuits(
            circuits,
            backend,
            qjob_config=qjob_config,
            backend_options=backend_options,
            noise_config=noise_config,
            run_config=run_config,
        )
        jobs = [job]
        job_ids = [job_id]
        split_circuits = [circuits]
    results = []
    if with_autorecover:
        logger.info("Backend status: %s", backend.status())
        logger.info("There are %s jobs are submitted.", len(jobs))
        logger.info("All job ids:\n%s", job_ids)
        for idx, _ in enumerate(jobs):
            result = None
            logger.info("Backend status: %s", backend.status())
            logger.info("There is one jobs are submitted: id: %s", job_id)
            job = jobs[idx]
            job_id = job_ids[idx]
            while True:
                logger.info("Running job id: %s", job_id)
                # try to get result if possible
                while True:
                    job_status = _safe_get_job_status(job, job_id)
                    queue_position = 0
                    if job_status in JOB_FINAL_STATES:
                        # do callback again after the job is in the final states
                        if job_callback is not None:
                            job_callback(job_id, job_status, queue_position, job)
                        break
                    if job_status == JobStatus.QUEUED and hasattr(job, "queue_position"):
                        queue_position = job.queue_position()
                        logger.info("Job id: %s is queued at position %s", job_id, queue_position)
                    else:
                        logger.info("Job id: %s, status: %s", job_id, job_status)
                    if job_callback is not None:
                        job_callback(job_id, job_status, queue_position, job)
                    time.sleep(qjob_config["wait"])

                # get result after the status is DONE
                if job_status == JobStatus.DONE:
                    while True:
                        result = job.result()
                        if result.success:
                            results.append(result)
                            logger.info("COMPLETED the %s-th job, job id: %s", idx, job_id)
                            break

                        logger.warning("FAILURE: Job id: %s", job_id)
                        logger.warning(
                            "Job (%s) is completed anyway, retrieve result " "from backend again.",
                            job_id,
                        )
                        job = backend.retrieve_job(job_id)
                    break
                # for other cases, resubmit the circuit until the result is available.
                # since if there is no result returned, there is no way algorithm can do any process
                if job_status == JobStatus.CANCELLED:
                    logger.warning(
                        "FAILURE: Job id: %s is cancelled. Re-submit the circuits.", job_id
                    )
                elif job_status == JobStatus.ERROR:
                    logger.warning(
                        "FAILURE: Job id: %s encounters the error. "
                        "Error is : %s. Re-submit the circuits.",
                        job_id,
                        job.error_message(),
                    )
                else:
                    logging.warning(
                        "FAILURE: Job id: %s. Unknown status: %s. " "Re-submit the circuits.",
                        job_id,
                        job_status,
                    )

                job, job_id = _safe_submit_circuits(
                    split_circuits[idx],
                    backend,
                    qjob_config=qjob_config,
                    backend_options=backend_options,
                    noise_config=noise_config,
                    run_config=run_config,
                )
    else:
        results = []
        for job in jobs:
            results.append(job.result())

    result = _combine_result_objects(results) if results else None

    # If result was not successful then raise an exception with either the status msg or
    # extra information if this was an Aer partial result return
    if not result.success:
        msg = result.status
        if result.status == "PARTIAL COMPLETED":
            # Aer can return partial results which Aqua algorithms cannot process and signals
            # using partial completed status where each returned result has a success and status.
            # We use the status from the first result that was not successful
            for res in result.results:
                if not res.success:
                    msg += ", " + res.status
                    break
        raise QiskitError(f"Circuit execution failed: {msg}")

    if not hasattr(result, "time_taken"):
        setattr(result, "time_taken", 0.0)

    return result


def _safe_submit_circuits(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Union[Backend, BaseBackend],
    qjob_config: Dict,
    backend_options: Dict,
    noise_config: Dict,
    run_config: Dict,
) -> Tuple[BaseJob, str]:
    # assure get job ids
    while True:
        try:
            job = _run_circuits_on_backend(
                backend,
                circuits,
                backend_options=backend_options,
                noise_config=noise_config,
                run_config=run_config,
            )
            job_id = job.job_id()
            break
        except QiskitError as ex:
            failure_warn = True
            if is_ibmq_provider(backend):
                try:
                    from qiskit.providers.ibmq import IBMQBackendJobLimitError
                except ImportError as ex1:
                    raise MissingOptionalLibraryError(
                        libname="qiskit-ibmq-provider",
                        name="_safe_submit_circuits",
                        pip_install="pip install qiskit-ibmq-provider",
                    ) from ex1
                if isinstance(ex, IBMQBackendJobLimitError):

                    oldest_running = backend.jobs(
                        limit=1, descending=False, status=["QUEUED", "VALIDATING", "RUNNING"]
                    )
                    if oldest_running:
                        oldest_running = oldest_running[0]
                        logger.warning(
                            "Job limit reached, waiting for job %s to finish "
                            "before submitting the next one.",
                            oldest_running.job_id(),
                        )
                        failure_warn = False  # Don't issue a second warning.
                        try:
                            oldest_running.wait_for_final_state(
                                timeout=qjob_config["timeout"], wait=qjob_config["wait"]
                            )
                        except Exception:  # pylint: disable=broad-except
                            # If the wait somehow fails or times out, we'll just re-try
                            # the job submit and see if it works now.
                            pass
            if failure_warn:
                logger.warning(
                    "FAILURE: Can not get job id, Resubmit the qobj to get job id. "
                    "Terra job error: %s ",
                    ex,
                )
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "FAILURE: Can not get job id, Resubmit the qobj to get job id." "Error: %s ", ex
            )

    return job, job_id


def _run_circuits_on_backend(
    backend: Union[Backend, BaseBackend],
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend_options: Dict,
    noise_config: Dict,
    run_config: Dict,
) -> BaseJob:
    """run on backend"""
    run_kwargs = {}
    if is_aer_provider(backend) or is_basicaer_provider(backend):
        for key, value in backend_options.items():
            if key == "backend_options":
                for k, v in value.items():
                    run_kwargs[k] = v
            else:
                run_kwargs[key] = value
    else:
        run_kwargs.update(backend_options)

    run_kwargs.update(noise_config)
    run_kwargs.update(run_config)

    if is_basicaer_provider(backend):
        # BasicAer emits warning if option is not in its list
        for key in list(run_kwargs.keys()):
            if not hasattr(backend.options, key):
                del run_kwargs[key]

    return backend.run(circuits, **run_kwargs)
