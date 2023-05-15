# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Measurement error mitigation"""

import copy
from typing import List, Optional, Tuple, Dict, Callable

from qiskit import compiler
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit.qobj import QasmQobj
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.utils.mitigation import (
    complete_meas_cal,
    tensored_meas_cal,
    CompleteMeasFitter,
    TensoredMeasFitter,
)
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def get_measured_qubits(
    transpiled_circuits: List[QuantumCircuit],
) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Deprecated: Retrieve the measured qubits from transpiled circuits.

    Args:
        transpiled_circuits: a list of transpiled circuits

    Returns:
        The used and sorted qubit index
        Key is qubit index str connected by '_',
              value is the experiment index. {str: list[int]}
    Raises:
        QiskitError: invalid qubit mapping
    """
    qubit_index = None
    qubit_mappings = {}
    for idx, qc in enumerate(transpiled_circuits):
        measured_qubits = []
        for instruction in qc.data:
            if instruction.operation.name != "measure":
                continue
            for qreg in qc.qregs:
                if instruction.qubits[0] in qreg:
                    index = qreg[:].index(instruction.qubits[0])
                    measured_qubits.append(index)
                    break
        measured_qubits_str = "_".join([str(x) for x in measured_qubits])
        if measured_qubits_str not in qubit_mappings:
            qubit_mappings[measured_qubits_str] = []
        qubit_mappings[measured_qubits_str].append(idx)
        if qubit_index is None:
            qubit_index = measured_qubits
        elif set(qubit_index) != set(measured_qubits):
            raise QiskitError(
                "The used qubit index are different. ({}) vs ({}).\nCurrently, "
                "we only support all circuits using the same set of qubits "
                "regardless qubit order.".format(qubit_index, measured_qubits)
            )

    return sorted(qubit_index), qubit_mappings


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def get_measured_qubits_from_qobj(qobj: QasmQobj) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Deprecated: Retrieve the measured qubits from transpiled circuits.

    Args:
        qobj: qobj

    Returns:
        the used and sorted qubit index
        key is qubit index str connected by '_',
              value is the experiment index. {str: list[int]}
     Raises:
        QiskitError: invalid qubit mapping
    """

    qubit_index = None
    qubit_mappings = {}

    for idx, exp in enumerate(qobj.experiments):
        measured_qubits = []
        for instr in exp.instructions:
            if instr.name != "measure":
                continue
            measured_qubits.append(instr.qubits[0])
        measured_qubits_str = "_".join([str(x) for x in measured_qubits])
        if measured_qubits_str not in qubit_mappings:
            qubit_mappings[measured_qubits_str] = []
        qubit_mappings[measured_qubits_str].append(idx)
        if qubit_index is None:
            qubit_index = measured_qubits
        else:
            if set(qubit_index) != set(measured_qubits):
                raise QiskitError(
                    "The used qubit index are different. ({}) vs ({}).\nCurrently, "
                    "we only support all circuits using the same set of qubits "
                    "regardless qubit order.".format(qubit_index, measured_qubits)
                )

    return sorted(qubit_index), qubit_mappings


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def build_measurement_error_mitigation_circuits(
    qubit_list: List[int],
    fitter_cls: Callable,
    backend: Backend,
    backend_config: Optional[Dict] = None,
    compile_config: Optional[Dict] = None,
    mit_pattern: Optional[List[List[int]]] = None,
) -> Tuple[QuantumCircuit, List[str], List[str]]:
    """Deprecated: Build measurement error mitigation circuits
    Args:
        qubit_list: list of ordered qubits used in the algorithm
        fitter_cls: CompleteMeasFitter or TensoredMeasFitter
        backend: backend instance
        backend_config: configuration for backend
        compile_config: configuration for compilation
        mit_pattern: Qubits on which to perform the
            measurement correction, divided to groups according to tensors.
            If `None` and `qr` is given then assumed to be performed over the entire
            `qr` as one group (default `None`).

    Returns:
        the circuit
        the state labels for build MeasFitter
        the labels of the calibration circuits
    Raises:
        QiskitError: when the fitter_cls is not recognizable.
    """
    circlabel = "mcal"

    if not qubit_list:
        raise QiskitError("The measured qubit list can not be [].")

    run = False
    if fitter_cls == CompleteMeasFitter:
        meas_calibs_circuits, state_labels = complete_meas_cal(
            qubit_list=range(len(qubit_list)), circlabel=circlabel
        )
        run = True
    elif fitter_cls == TensoredMeasFitter:
        meas_calibs_circuits, state_labels = tensored_meas_cal(
            mit_pattern=mit_pattern, circlabel=circlabel
        )
        run = True
    if not run:
        try:
            from qiskit.ignis.mitigation.measurement import (
                CompleteMeasFitter as CompleteMeasFitter_IG,
                TensoredMeasFitter as TensoredMeasFitter_IG,
            )
        except ImportError as ex:
            # If ignis can't be imported we don't have a valid fitter
            # class so just fail here with an appropriate error message
            raise QiskitError(f"Unknown fitter {fitter_cls}") from ex
        if fitter_cls == CompleteMeasFitter_IG:
            meas_calibs_circuits, state_labels = complete_meas_cal(
                qubit_list=range(len(qubit_list)), circlabel=circlabel
            )
        elif fitter_cls == TensoredMeasFitter_IG:
            meas_calibs_circuits, state_labels = tensored_meas_cal(
                mit_pattern=mit_pattern, circlabel=circlabel
            )
        else:
            raise QiskitError(f"Unknown fitter {fitter_cls}")

    # the provided `qubit_list` would be used as the initial layout to
    # assure the consistent qubit mapping used in the main circuits.

    tmp_compile_config = copy.deepcopy(compile_config)
    tmp_compile_config["initial_layout"] = qubit_list
    t_meas_calibs_circuits = compiler.transpile(
        meas_calibs_circuits, backend, **backend_config, **tmp_compile_config
    )
    return t_meas_calibs_circuits, state_labels, circlabel


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def build_measurement_error_mitigation_qobj(
    qubit_list: List[int],
    fitter_cls: Callable,
    backend: Backend,
    backend_config: Optional[Dict] = None,
    compile_config: Optional[Dict] = None,
    run_config: Optional[RunConfig] = None,
    mit_pattern: Optional[List[List[int]]] = None,
) -> Tuple[QasmQobj, List[str], List[str]]:
    """
    Args:
        qubit_list: list of ordered qubits used in the algorithm
        fitter_cls: CompleteMeasFitter or TensoredMeasFitter
        backend: backend instance
        backend_config: configuration for backend
        compile_config: configuration for compilation
        run_config: configuration for running a circuit
        mit_pattern: Qubits on which to perform the
            measurement correction, divided to groups according to tensors.
            If `None` and `qr` is given then assumed to be performed over the entire
            `qr` as one group (default `None`).

    Returns:
        the Qobj with calibration circuits at the beginning
        the state labels for build MeasFitter
        the labels of the calibration circuits

    Raises:
        QiskitError: when the fitter_cls is not recognizable.
        MissingOptionalLibraryError: Qiskit-Ignis not installed
    """

    circlabel = "mcal"

    if not qubit_list:
        raise QiskitError("The measured qubit list can not be [].")

    if fitter_cls == CompleteMeasFitter:
        meas_calibs_circuits, state_labels = complete_meas_cal(
            qubit_list=range(len(qubit_list)), circlabel=circlabel
        )
    elif fitter_cls == TensoredMeasFitter:
        meas_calibs_circuits, state_labels = tensored_meas_cal(
            mit_pattern=mit_pattern, circlabel=circlabel
        )
    else:
        raise QiskitError(f"Unknown fitter {fitter_cls}")

    # the provided `qubit_list` would be used as the initial layout to
    # assure the consistent qubit mapping used in the main circuits.

    tmp_compile_config = copy.deepcopy(compile_config)
    tmp_compile_config["initial_layout"] = qubit_list
    t_meas_calibs_circuits = compiler.transpile(
        meas_calibs_circuits, backend, **backend_config, **tmp_compile_config
    )
    cals_qobj = compiler.assemble(t_meas_calibs_circuits, backend, **run_config.to_dict())
    if hasattr(cals_qobj.config, "parameterizations"):
        del cals_qobj.config.parameterizations
    return cals_qobj, state_labels, circlabel
