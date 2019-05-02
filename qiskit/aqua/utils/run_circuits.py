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

import sys
import logging
import time
import copy
import os
import uuid

import numpy as np
from qiskit import compiler
from qiskit.assembler import assemble_circuits
from qiskit.providers import BaseBackend, JobStatus, JobError
from qiskit.providers.basicaer import BasicAerJob
from qiskit.qobj import QobjHeader
from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua.utils import summarize_circuits
from qiskit.aqua.utils.backend_utils import (is_aer_provider,
                                             is_basicaer_provider,
                                             is_ibmq_provider,
                                             is_simulator_backend,
                                             is_local_backend)

MAX_CIRCUITS_PER_JOB = os.environ.get('QISKIT_AQUA_MAX_CIRCUITS_PER_JOB', None)

logger = logging.getLogger(__name__)


def find_regs_by_name(circuit, name, qreg=True):
    """Find the registers in the circuits.

    Args:
        circuit (QuantumCircuit): the quantum circuit.
        name (str): name of register
        qreg (bool): quantum or classical register

    Returns:
        QuantumRegister or ClassicalRegister or None: if not found, return None.

    """
    found_reg = None
    regs = circuit.qregs if qreg else circuit.cregs
    for reg in regs:
        if reg.name == name:
            found_reg = reg
            break
    return found_reg


def _avoid_empty_circuits(circuits):
    new_circuits = []
    for qc in circuits:
        if len(qc) == 0:
            tmp_q = None
            for q in qc.qregs:
                tmp_q = q
                break
            if tmp_q is None:
                raise NameError("A QASM without any quantum register is invalid.")
            qc.iden(tmp_q[0])
        new_circuits.append(qc)
    return new_circuits


def _combine_result_objects(results):
    """Tempoary helper function.

    TODO:
        This function would be removed after Terra supports job with infinite circuits.
    """
    if len(results) == 1:
        return results[0]

    new_result = copy.deepcopy(results[0])

    for idx in range(1, len(results)):
        new_result.results.extend(results[idx].results)

    return new_result


def _maybe_add_aer_expectation_instruction(qobj, options):
    if 'expectation' in options:
        from qiskit.providers.aer.utils.qobj_utils import snapshot_instr, append_instr, get_instr_pos
        # add others, how to derive the correct used number of qubits?
        # the compiled qobj could be wrong if coupling map is used.
        params = options['expectation']['params']
        num_qubits = options['expectation']['num_qubits']

        for idx in range(len(qobj.experiments)):
            # if mulitple params are provided, we assume that each circuit is corresponding one param
            # otherwise, params are used for all circuits.
            param_idx = idx if len(params) > 1 else 0
            snapshot_pos = get_instr_pos(qobj, idx, 'snapshot')
            if len(snapshot_pos) == 0:  # does not append the instruction yet.
                new_ins = snapshot_instr('expectation_value_pauli', 'test',
                                         list(range(num_qubits)), params=params[param_idx])
                qobj = append_instr(qobj, idx, new_ins)
            else:
                for i in snapshot_pos:  # update all expectation_value_snapshot
                    if qobj.experiments[idx].instructions[i].type == 'expectation_value_pauli':
                        qobj.experiments[idx].instructions[i].params = params[param_idx]
    return qobj


def _compile_wrapper(circuits, backend, backend_config, compile_config, run_config):
    transpiled_circuits = compiler.transpile(circuits, backend, **backend_config, **compile_config)
    if not isinstance(transpiled_circuits, list):
        transpiled_circuits = [transpiled_circuits]

    qobj = assemble_circuits(transpiled_circuits, qobj_id=str(uuid.uuid4()), qobj_header=QobjHeader(),
                             run_config=run_config)
    return qobj, transpiled_circuits


def compile_circuits(circuits, backend, backend_config=None, compile_config=None, run_config=None,
                     show_circuit_summary=False, circuit_cache=None, **kwargs):
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The autorecovery feature is only applied for non-simulator backend.
    This wraper will try to get the result no matter how long it costs.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): backend instance
        backend_config (dict, optional): configuration for backend
        compile_config (dict, optional): configuration for compilation
        run_config (RunConfig, optional): configuration for running a circuit
        show_circuit_summary (bool, optional): showing the summary of submitted circuits.
        circuit_cache (CircuitCache, optional): A CircuitCache to use when calling compile_and_run_circuits

    Returns:
        QasmObj: compiled qobj.

    Raises:
        AquaError: Any error except for JobError raised by Qiskit Terra
    """
    backend_config = backend_config or {}
    compile_config = compile_config or {}
    run_config = run_config or {}

    if backend is None or not isinstance(backend, BaseBackend):
        raise ValueError('Backend is missing or not an instance of BaseBackend')

    if not isinstance(circuits, list):
        circuits = [circuits]

    if is_simulator_backend(backend):
        circuits = _avoid_empty_circuits(circuits)

    if MAX_CIRCUITS_PER_JOB is not None:
        max_circuits_per_job = int(MAX_CIRCUITS_PER_JOB)
    else:
        if is_local_backend(backend):
            max_circuits_per_job = sys.maxsize
        else:
            max_circuits_per_job = backend.configuration().max_experiments

    if circuit_cache is not None and circuit_cache.try_reusing_qobjs:
        # Check if all circuits are the same length.
        # If not, don't try to use the same qobj.experiment for all of them.
        if len(set([len(circ.data) for circ in circuits])) > 1:
            circuit_cache.try_reusing_qobjs = False
        else:  # Try setting up the reusable qobj
            # Compile and cache first circuit if cache is empty. The load method will try to reuse it
            if circuit_cache.qobjs is None:
                qobj, transpiled_circuits = _compile_wrapper([circuits[0]], backend, backend_config,
                                                             compile_config, run_config)

                if is_aer_provider(backend):
                    qobj = _maybe_add_aer_expectation_instruction(qobj, kwargs)
                circuit_cache.cache_circuit(qobj, [circuits[0]], 0)

    qobjs = []
    transpiled_circuits = []
    chunks = int(np.ceil(len(circuits) / max_circuits_per_job))
    for i in range(chunks):
        sub_circuits = circuits[i * max_circuits_per_job:(i + 1) * max_circuits_per_job]
        if circuit_cache is not None and circuit_cache.misses < circuit_cache.allowed_misses:
            try:
                if circuit_cache.cache_transpiled_circuits:
                    transpiled_sub_circuits = compiler.transpile(sub_circuits, backend, **backend_config,
                                                                 **compile_config)
                    qobj = circuit_cache.load_qobj_from_cache(transpiled_sub_circuits, i, run_config=run_config)
                else:
                    qobj = circuit_cache.load_qobj_from_cache(sub_circuits, i, run_config=run_config)
                if is_aer_provider(backend):
                    qobj = _maybe_add_aer_expectation_instruction(qobj, kwargs)
            # cache miss, fail gracefully
            except (TypeError, IndexError, FileNotFoundError, EOFError, AquaError, AttributeError) as e:
                circuit_cache.try_reusing_qobjs = False  # Reusing Qobj didn't work
                if len(circuit_cache.qobjs) > 0:
                    logger.info('Circuit cache miss, recompiling. Cache miss reason: ' + repr(e))
                    circuit_cache.misses += 1
                else:
                    logger.info('Circuit cache is empty, compiling from scratch.')
                circuit_cache.clear_cache()
                qobj, transpiled_sub_circuits = _compile_wrapper(sub_circuits, backend, backend_config,
                                                                 compile_config, run_config)
                transpiled_circuits.extend(transpiled_sub_circuits)
                if is_aer_provider(backend):
                    qobj = _maybe_add_aer_expectation_instruction(qobj, kwargs)
                try:
                    circuit_cache.cache_circuit(qobj, sub_circuits, i)
                except (TypeError, IndexError, AquaError, AttributeError, KeyError) as e:
                    try:
                        circuit_cache.cache_transpiled_circuits = True
                        circuit_cache.cache_circuit(qobj, transpiled_sub_circuits, i)
                    except (TypeError, IndexError, AquaError, AttributeError, KeyError) as e:
                        logger.info('Circuit could not be cached for reason: ' + repr(e))
                        logger.info('Transpilation may be too aggressive. Try skipping transpiler.')

        else:
            qobj, transpiled_sub_circuits = _compile_wrapper(sub_circuits, backend, backend_config, compile_config,
                                                             run_config)
            transpiled_circuits.extend(transpiled_sub_circuits)
            if is_aer_provider(backend):
                qobj = _maybe_add_aer_expectation_instruction(qobj, kwargs)

        qobjs.append(qobj)

    if logger.isEnabledFor(logging.DEBUG) and show_circuit_summary:
        logger.debug("==== Before transpiler ====")
        logger.debug(summarize_circuits(circuits))
        logger.debug("====  After transpiler ====")
        logger.debug(summarize_circuits(transpiled_circuits))

    return qobjs


def _safe_submit_qobj(qobj, backend, backend_options, noise_config, skip_qobj_validation):
    # assure get job ids
    while True:
        job = run_on_backend(backend, qobj, backend_options=backend_options, noise_config=noise_config,
                             skip_qobj_validation=skip_qobj_validation)
        try:
            job_id = job.job_id()
            break
        except JobError as e:
            logger.warning("FAILURE: Can not get job id, Resubmit the qobj to get job id."
                           "Terra job error: {} ".format(e))
        except Exception as e:
            logger.warning("FAILURE: Can not get job id, Resubmit the qobj to get job id."
                           "Error: {} ".format(e))

    return job, job_id


def run_qobjs(qobjs, backend, qjob_config=None, backend_options=None,
              noise_config=None, skip_qobj_validation=False):
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The autorecovery feature is only applied for non-simulator backend.
    This wraper will try to get the result no matter how long it costs.

    Args:
        qobjs (list[QasmObj]): qobjs to execute
        backend (BaseBackend): backend instance
        qjob_config (dict, optional): configuration for quantum job object
        backend_options (dict, optional): configuration for simulator
        noise_config (dict, optional): configuration for noise model
        skip_qobj_validation (bool, optional): Bypass Qobj validation to decrease submission time

    Returns:
        Result: Result object

    Raises:
        AquaError: Any error except for JobError raised by Qiskit Terra
    """
    qjob_config = qjob_config or {}
    backend_options = backend_options or {}
    noise_config = noise_config or {}

    if backend is None or not isinstance(backend, BaseBackend):
        raise ValueError('Backend is missing or not an instance of BaseBackend')

    with_autorecover = False if is_simulator_backend(backend) else True

    jobs = []
    job_ids = []
    for qobj in qobjs:
        job, job_id = _safe_submit_qobj(qobj, backend, backend_options, noise_config, skip_qobj_validation)
        job_ids.append(job_id)
        jobs.append(job)

    results = []
    if with_autorecover:
        logger.info("Backend status: {}".format(backend.status()))
        logger.info("There are {} jobs are submitted.".format(len(jobs)))
        logger.info("All job ids:\n{}".format(job_ids))
        for idx in range(len(jobs)):
            job = jobs[idx]
            job_id = job_ids[idx]
            while True:
                logger.info("Running {}-th qobj, job id: {}".format(idx, job_id))
                # try to get result if possible
                try:
                    result = job.result(**qjob_config)
                    if result.success:
                        results.append(result)
                        logger.info("COMPLETED the {}-th qobj, "
                                    "job id: {}".format(idx, job_id))
                        break
                    else:
                        logger.warning("FAILURE: the {}-th qobj, "
                                       "job id: {}".format(idx, job_id))
                except JobError as e:
                    # if terra raise any error, which means something wrong, re-run it
                    logger.warning("FAILURE: the {}-th qobj, job id: {} "
                                   "Terra job error: {} ".format(idx, job_id, e))
                except Exception as e:
                    raise AquaError("FAILURE: the {}-th qobj, job id: {} "
                                    "Unknown error: {} ".format(idx, job_id, e)) from e

                # something wrong here if reach here, querying the status to check how to handle it.
                # keep qeurying it until getting the status.
                while True:
                    try:
                        job_status = job.status()
                        break
                    except JobError as e:
                        logger.warning("FAILURE: job id: {}, "
                                       "status: 'FAIL_TO_GET_STATUS' "
                                       "Terra job error: {}".format(job_id, e))
                        time.sleep(5)
                    except Exception as e:
                        raise AquaError("FAILURE: job id: {}, "
                                        "status: 'FAIL_TO_GET_STATUS' "
                                        "Unknown error: ({})".format(job_id, e)) from e

                logger.info("Job status: {}".format(job_status))

                # handle the failure job based on job status
                if job_status == JobStatus.DONE:
                    logger.info("Job ({}) is completed anyway, retrieve result "
                                "from backend.".format(job_id))
                    job = backend.retrieve_job(job_id)
                elif job_status == JobStatus.RUNNING or job_status == JobStatus.QUEUED:
                    logger.info("Job ({}) is {}, but encounter an exception, "
                                "recover it from backend.".format(job_id, job_status))
                    job = backend.retrieve_job(job_id)
                else:
                    logger.info("Fail to run Job ({}), resubmit it.".format(job_id))
                    qobj = qobjs[idx]
                    #  assure job get its id
                    job, job_id = _safe_submit_qobj(qobj, backend, backend_options, noise_config, skip_qobj_validation)
                    jobs[idx] = job
                    job_ids[idx] = job_id
    else:
        results = []
        for job in jobs:
            results.append(job.result(**qjob_config))

    result = _combine_result_objects(results) if len(results) != 0 else None

    return result


def compile_and_run_circuits(circuits, backend, backend_config=None,
                             compile_config=None, run_config=None,
                             qjob_config=None, backend_options=None,
                             noise_config=None, show_circuit_summary=False,
                             circuit_cache=None, skip_qobj_validation=False, **kwargs):
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The autorecovery feature is only applied for non-simulator backend.
    This wraper will try to get the result no matter how long it costs.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): backend instance
        backend_config (dict, optional): configuration for backend
        compile_config (dict, optional): configuration for compilation
        run_config (RunConfig, optional): configuration for running a circuit
        qjob_config (dict, optional): configuration for quantum job object
        backend_options (dict, optional): configuration for simulator
        noise_config (dict, optional): configuration for noise model
        show_circuit_summary (bool, optional): showing the summary of submitted circuits.
        circuit_cache (CircuitCache, optional): A CircuitCache to use when calling compile_and_run_circuits
        skip_qobj_validation (bool, optional): Bypass Qobj validation to decrease submission time

    Returns:
        Result: Result object

    Raises:
        AquaError: Any error except for JobError raised by Qiskit Terra
    """
    qobjs = compile_circuits(circuits, backend, backend_config, compile_config, run_config,
                             show_circuit_summary, circuit_cache, **kwargs)
    result = run_qobjs(qobjs, backend, qjob_config, backend_options, noise_config, skip_qobj_validation)
    return result


# skip_qobj_validation = True does what backend.run and aerjob.submit do, but without qobj validation.
def run_on_backend(backend, qobj, backend_options=None, noise_config=None, skip_qobj_validation=False):
    if skip_qobj_validation:
        job_id = str(uuid.uuid4())
        if is_aer_provider(backend):
            from qiskit.providers.aer.aerjob import AerJob
            temp_backend_options = backend_options['backend_options'] if backend_options != {} else None
            temp_noise_config = noise_config['noise_model'] if noise_config != {} else None
            job = AerJob(backend, job_id, backend._run_job, qobj, temp_backend_options, temp_noise_config, False)
            job._future = job._executor.submit(job._fn, job._job_id, job._qobj, *job._args)
        elif is_basicaer_provider(backend):
            backend._set_options(qobj_config=qobj.config, **backend_options)
            job = BasicAerJob(backend, job_id, backend._run_job, qobj)
            job._future = job._executor.submit(job._fn, job._job_id, job._qobj)
        elif is_ibmq_provider(backend):
            # TODO: IBMQJob performs validation during the constructor. the following lines does not
            # skip validation but run as is.
            from qiskit.providers.ibmq.ibmqjob import IBMQJob
            job = IBMQJob(backend, None, backend._api, qobj=qobj)
            job._future = job._executor.submit(job._submit_callback)
        else:
            logger.info("Can't skip qobj validation for the third-party provider.")
            job = backend.run(qobj, **backend_options, **noise_config)
        return job
    else:
        job = backend.run(qobj, **backend_options, **noise_config)
        return job
