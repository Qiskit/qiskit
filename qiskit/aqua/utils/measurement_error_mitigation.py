# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)

from .run_circuits import run_qobjs, compile_circuits
from ..aqua_error import AquaError

logger = logging.getLogger(__name__)


def get_measured_qubits(transpiled_circuits):
    """
    Retrieve the measured qubits from transpiled circuits.

    Args:
        transpiled_circuits ([QuantumCircuit]): a list of transpiled circuits

    Returns:
        [int]: the qubit mapping to-be-used for measure error mitigation
    """

    qubit_mapping = None
    for qc in transpiled_circuits:
        measured_qubits = []
        for inst, qargs, cargs in qc.data:
            if inst.name != 'measure':
                continue
            measured_qubits.append(qargs[0][1])
        if qubit_mapping is None:
            qubit_mapping = measured_qubits
        elif qubit_mapping != measured_qubits:
            raise AquaError("The qubit mapping of circuits are different."
                            "Currently, we only support single mapping.")

    return qubit_mapping


def get_measured_qubits_from_qobj(qobjs):
    """
    Retrieve the measured qubits from transpiled circuits.

    Args:
        qobjs (list[QasmObj]): qobjs

    Returns:
        [int]: the qubit mapping to-be-used for measure error mitigation
    """

    qubit_mapping = None
    for qobj in qobjs:
        for exp in qobj.experiments:
            measured_qubits = []
            for instr in exp.instructions:
                if instr.name != 'measure':
                    continue
                measured_qubits.append(instr.qubits[0])
            if qubit_mapping is None:
                qubit_mapping = measured_qubits
            else:
                if qubit_mapping != measured_qubits:
                    raise AquaError("The qubit mapping of circuits are different."
                                    "Currently, we only support single mapping.")

    return qubit_mapping


def build_measurement_error_mitigation_fitter(qubits, fitter_cls, backend,
                                              backend_config=None, compile_config=None,
                                              run_config=None, qjob_config=None, backend_options=None,
                                              noise_config=None):
    """

    Args:
        qubits (list[int]): the measured qubit index (in the order to classical bit 0...n-1)
        fitter_cls (callable): CompleteMeasFitter or TensoredMeasFitter
        backend (BaseBackend): backend instance
        backend_config (dict, optional): configuration for backend
        compile_config (dict, optional): configuration for compilation
        run_config (RunConfig, optional): configuration for running a circuit
        qjob_config (dict, optional): configuration for quantum job object
        backend_options (dict, optional): configuration for simulator
        noise_config (dict, optional): configuration for noise model

    Returns:
        CompleteMeasFitter or TensoredMeasFitter: the measurement fitter

    Raises:
        AquaError: when the fitter_cls is not recognizable.
    """

    if len(qubits) == 0:
        raise AquaError("The measured qubits can not be [].")

    circlabel = 'mcal'

    if fitter_cls == CompleteMeasFitter:
        meas_calibs_circuits, state_labels = complete_meas_cal(qubit_list=qubits, circlabel=circlabel)
    elif fitter_cls == TensoredMeasFitter:
        # TODO support different calibration
        raise AquaError("Does not support TensoredMeasFitter yet.")
        # meas_calibs_circuits, state_labels = tensored_meas_cal()
    else:
        raise AquaError("Unknown fitter {}".format(fitter_cls))

    # compile
    qobjs = compile_circuits(meas_calibs_circuits, backend, backend_config, compile_config, run_config)
    cal_results = run_qobjs(qobjs, backend, qjob_config, backend_options, noise_config,
                            skip_qobj_validation=False)
    meas_fitter = fitter_cls(cal_results, state_labels, circlabel=circlabel)

    if fitter_cls == CompleteMeasFitter:
        logger.info("Calibration matrix:\n{}".format(meas_fitter.cal_matrix))
    elif fitter_cls == TensoredMeasFitter:
        logger.info("Calibration matrices:\n{}".format(meas_fitter.cal_matrices))

    return meas_fitter


def mitigate_measurement_error(results, meas_fitter, method='least_squares'):
    """

    Args:
        results (Result): the unmitigated Result object
        meas_fitter (CompleteMeasFitter or TensoredMeasFitter): the measurement fitter
        method (str): fitting method. If None, then least_squares is used.
                'pseudo_inverse': direct inversion of the A matrix
                'least_squares': constrained to have physical probabilities
    Returns:
        Result: the mitigated Result
    """
    # Get the filter object
    meas_filter = meas_fitter.filter

    # Results without mitigation
    mitigated_results = meas_filter.apply(results, method=method)

    return mitigated_results
