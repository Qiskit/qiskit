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

""" Measurement error mitigation """

import copy
import logging

from qiskit import compiler
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)

from ..aqua_error import AquaError

logger = logging.getLogger(__name__)


def get_measured_qubits(transpiled_circuits):
    """
    Retrieve the measured qubits from transpiled circuits.

    Args:
        transpiled_circuits ([QuantumCircuit]): a list of transpiled circuits

    Returns:
        list[int]: the qubit mapping to-be-used for measure error mitigation
    Raises:
        AquaError: invalid qubit mapping
    """

    qubit_mapping = None
    for qc in transpiled_circuits:
        measured_qubits = []
        for inst, qargs, _ in qc.data:
            if inst.name != 'measure':
                continue
            measured_qubits.append(qargs[0][1])
        if qubit_mapping is None:
            qubit_mapping = measured_qubits
        elif qubit_mapping != measured_qubits:
            raise AquaError("The qubit mapping of circuits are different."
                            "Currently, we only support single mapping.")

    return qubit_mapping


def get_measured_qubits_from_qobj(qobj):
    """
    Retrieve the measured qubits from transpiled circuits.

    Args:
        qobj (QasmObj): qobj

    Returns:
        list[int]: the qubit mapping to-be-used for measure error mitigation
     Raises:
        AquaError: invalid qubit mapping
    """

    qubit_mapping = None

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


# pylint: disable=invalid-name
def build_measurement_error_mitigation_qobj(qubit_list, fitter_cls, backend,
                                            backend_config=None, compile_config=None,
                                            run_config=None):
    """
        Args:
            qubit_list (list[int]): list of ordered qubits used in the algorithm
            fitter_cls (callable): CompleteMeasFitter or TensoredMeasFitter
            backend (BaseBackend): backend instance
            backend_config (dict, optional): configuration for backend
            compile_config (dict, optional): configuration for compilation
            run_config (RunConfig, optional): configuration for running a circuit

        Returns:
            QasmQobj: the Qobj with calibration circuits at the beginning
            list[str]: the state labels for build MeasFitter
            list[str]: the labels of the calibration circuits

        Raises:
            AquaError: when the fitter_cls is not recognizable.
        """

    circlabel = 'mcal'

    if not qubit_list:
        raise AquaError("The measured qubit list can not be [].")

    if fitter_cls == CompleteMeasFitter:
        meas_calibs_circuits, state_labels = \
            complete_meas_cal(qubit_list=range(len(qubit_list)), circlabel=circlabel)
    elif fitter_cls == TensoredMeasFitter:
        # TODO support different calibration
        raise AquaError("Does not support TensoredMeasFitter yet.")
    else:
        raise AquaError("Unknown fitter {}".format(fitter_cls))

    # the provided `qubit_list` would be used as the initial layout to
    # assure the consistent qubit mapping used in the main circuits.

    tmp_compile_config = copy.deepcopy(compile_config)
    tmp_compile_config['initial_layout'] = qubit_list
    t_meas_calibs_circuits = compiler.transpile(meas_calibs_circuits, backend,
                                                **backend_config, **tmp_compile_config)
    cals_qobj = compiler.assemble(t_meas_calibs_circuits, backend, **run_config.to_dict())
    return cals_qobj, state_labels, circlabel
