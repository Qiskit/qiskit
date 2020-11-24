# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assemble function for converting a list of circuits into a qobj."""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (QasmQobj, QobjExperimentHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment,
                         QasmQobjConfig, QasmExperimentCalibrations, GateCalibration,
                         PulseQobjInstruction, PulseLibraryItem, converters, QobjHeader)
from qiskit.tools.parallel import parallel_map


PulseLibrary = Dict[str, List[complex]]


def _assemble_circuit(
        circuit: QuantumCircuit,
        run_config: RunConfig
) -> Tuple[QasmQobjExperiment, Optional[PulseLibrary]]:
    """Assemble one circuit.

    Args:
        circuit: circuit to assemble
        run_config: configuration of the runtime environment

    Returns:
        One experiment for the QasmQobj, and pulse library for pulse gates (which could be None)

    Raises:
        QiskitError: when the circuit has unit other than 'dt'.
    """
    if circuit.unit != 'dt':
        raise QiskitError("Unable to assemble circuit with unit '{}', which must be 'dt'."
                          .format(circuit.unit))

    # header data
    num_qubits = 0
    memory_slots = 0
    qubit_labels = []
    clbit_labels = []

    qreg_sizes = []
    creg_sizes = []
    for qreg in circuit.qregs:
        qreg_sizes.append([qreg.name, qreg.size])
        for j in range(qreg.size):
            qubit_labels.append([qreg.name, j])
        num_qubits += qreg.size
    for creg in circuit.cregs:
        creg_sizes.append([creg.name, creg.size])
        for j in range(creg.size):
            clbit_labels.append([creg.name, j])
        memory_slots += creg.size

    # TODO: why do we need creq_sizes and qreg_sizes in header
    # TODO: we need to rethink memory_slots as they are tied to classical bit
    header = QobjExperimentHeader(qubit_labels=qubit_labels,
                                  n_qubits=num_qubits,
                                  qreg_sizes=qreg_sizes,
                                  clbit_labels=clbit_labels,
                                  memory_slots=memory_slots,
                                  creg_sizes=creg_sizes,
                                  name=circuit.name,
                                  global_phase=float(circuit.global_phase))

    # TODO: why do we need n_qubits and memory_slots in both the header and the config
    config = QasmQobjExperimentConfig(n_qubits=num_qubits, memory_slots=memory_slots)

    calibrations, pulse_library = _assemble_pulse_gates(circuit, run_config)
    if calibrations:
        config.calibrations = calibrations

    # Convert conditionals from QASM-style (creg ?= int) to qobj-style
    # (register_bit ?= 1), by assuming device has unlimited register slots
    # (supported only for simulators). Map all measures to a register matching
    # their clbit_index, create a new register slot for every conditional gate
    # and add a bfunc to map the creg=val mask onto the gating register bit.

    is_conditional_experiment = any(op.condition for (op, qargs, cargs) in circuit.data)
    max_conditional_idx = 0

    instructions = []
    for op_context in circuit.data:
        instruction = op_context[0].assemble()

        # Add register attributes to the instruction
        qargs = op_context[1]
        cargs = op_context[2]
        if qargs:
            qubit_indices = [qubit_labels.index([qubit.register.name, qubit.index])
                             for qubit in qargs]
            instruction.qubits = qubit_indices
        if cargs:
            clbit_indices = [clbit_labels.index([clbit.register.name, clbit.index])
                             for clbit in cargs]
            instruction.memory = clbit_indices
            # If the experiment has conditional instructions, assume every
            # measurement result may be needed for a conditional gate.
            if instruction.name == "measure" and is_conditional_experiment:
                instruction.register = clbit_indices

        # To convert to a qobj-style conditional, insert a bfunc prior
        # to the conditional instruction to map the creg ?= val condition
        # onto a gating register bit.
        if hasattr(instruction, '_condition'):
            ctrl_reg, ctrl_val = instruction._condition
            mask = 0
            val = 0
            for clbit in clbit_labels:
                if clbit[0] == ctrl_reg.name:
                    mask |= (1 << clbit_labels.index(clbit))
                    val |= (((ctrl_val >> clbit[1]) & 1) << clbit_labels.index(clbit))

            conditional_reg_idx = memory_slots + max_conditional_idx
            conversion_bfunc = QasmQobjInstruction(name='bfunc',
                                                   mask="0x%X" % mask,
                                                   relation='==',
                                                   val="0x%X" % val,
                                                   register=conditional_reg_idx)
            instructions.append(conversion_bfunc)
            instruction.conditional = conditional_reg_idx
            max_conditional_idx += 1
            # Delete condition attribute now that we have replaced it with
            # the conditional and bfuc
            del instruction._condition

        instructions.append(instruction)
    return (QasmQobjExperiment(instructions=instructions, header=header, config=config),
            pulse_library)


def _assemble_pulse_gates(
        circuit: QuantumCircuit,
        run_config: RunConfig
) -> Tuple[Optional[QasmExperimentCalibrations], Optional[PulseLibrary]]:
    """Assemble and return the circuit calibrations and associated pulse library, if there are any.
    The calibrations themselves may reference the pulse library which is returned as a dict.

    Args:
        circuit: circuit which may have pulse calibrations
        run_config: configuration of the runtime environment

    Returns:
        The calibrations and pulse library, if there are any
    """
    if not circuit.calibrations:
        return None, None
    if not hasattr(run_config, 'parametric_pulses'):
        run_config.parametric_pulses = []
    calibrations = []
    pulse_library = {}
    for gate, cals in circuit.calibrations.items():
        for (qubits, params), schedule in cals.items():
            qobj_instructions, _ = _assemble_schedule(
                schedule,
                converters.InstructionToQobjConverter(PulseQobjInstruction),
                run_config,
                pulse_library)
            calibrations.append(
                GateCalibration(str(gate), list(qubits), list(params), qobj_instructions))
    return QasmExperimentCalibrations(gates=calibrations), pulse_library


def _extract_common_calibrations(
        experiments: List[QasmQobjExperiment]
) -> Tuple[List[QasmQobjExperiment], Optional[QasmExperimentCalibrations]]:
    """Given a list of ``QasmQobjExperiment``s, each of which may have calibrations in their
    ``config``, collect common calibrations into a global ``QasmExperimentCalibrations``
    and delete them from their local experiments.

    Args:
        experiments: The list of Qasm experiments that are being assembled into one qobj

    Returns:
        The input experiments with modified calibrations, and common calibrations, if there
        are any
    """
    def index_calibrations() -> Dict[int, List[Tuple[int, GateCalibration]]]:
        """Map each calibration to all experiments that contain it."""
        exp_indices = defaultdict(list)
        for exp_idx, exp in enumerate(experiments):
            for gate_cal in exp.config.calibrations.gates:
                # They must be keyed on the hash or identical cals will be indexed separately
                exp_indices[hash(gate_cal)].append((exp_idx, gate_cal))
        return exp_indices

    def collect_common_calibrations() -> List[GateCalibration]:
        """If a gate calibration appears in all experiments, collect it."""
        common_calibrations = []
        for _, exps_w_cal in exp_indices.items():
            if len(exps_w_cal) == len(experiments):
                _, gate_cal = exps_w_cal[0]
                common_calibrations.append(gate_cal)
        return common_calibrations

    def remove_common_gate_calibrations(exps: List[QasmQobjExperiment]) -> None:
        """For calibrations that appear in all experiments, remove them from the individual
        experiment's ``config.calibrations``."""
        for _, exps_w_cal in exp_indices.items():
            if len(exps_w_cal) == len(exps):
                for exp_idx, gate_cal in exps_w_cal:
                    exps[exp_idx].config.calibrations.gates.remove(gate_cal)

    if not (experiments and all(hasattr(exp.config, 'calibrations') for exp in experiments)):
        # No common calibrations
        return experiments, None

    exp_indices = index_calibrations()
    common_calibrations = collect_common_calibrations()
    remove_common_gate_calibrations(experiments)

    # Remove the ``calibrations`` attribute if it's now empty
    for exp in experiments:
        if not exp.config.calibrations.gates:
            del exp.config.calibrations

    return experiments, QasmExperimentCalibrations(gates=common_calibrations)


def assemble_circuits(
        circuits: List[QuantumCircuit],
        run_config: RunConfig,
        qobj_id: int,
        qobj_header: QobjHeader
) -> QasmQobj:
    """Assembles a list of circuits into a qobj that can be run on the backend.

    Args:
        circuits: circuit(s) to assemble
        run_config: configuration of the runtime environment
        qobj_id: identifier for the generated qobj
        qobj_header: header to pass to the results

    Returns:
        The qobj to be run on the backends
    """
    qobj_config = QasmQobjConfig()
    if run_config:
        qobj_config = QasmQobjConfig(**run_config.to_dict())
    qubit_sizes = []
    memory_slot_sizes = []
    for circ in circuits:
        num_qubits = 0
        memory_slots = 0
        for qreg in circ.qregs:
            num_qubits += qreg.size
        for creg in circ.cregs:
            memory_slots += creg.size
        qubit_sizes.append(num_qubits)
        memory_slot_sizes.append(memory_slots)
    qobj_config.memory_slots = max(memory_slot_sizes)
    qobj_config.n_qubits = max(qubit_sizes)

    experiments_and_pulse_libs = parallel_map(_assemble_circuit, circuits, [run_config])
    experiments = []
    pulse_library = {}
    for exp, lib in experiments_and_pulse_libs:
        experiments.append(exp)
        if lib:
            pulse_library.update(lib)
    if pulse_library:
        qobj_config.pulse_library = [PulseLibraryItem(name=name, samples=samples)
                                     for name, samples in pulse_library.items()]
    experiments, calibrations = _extract_common_calibrations(experiments)
    if calibrations and calibrations.gates:
        qobj_config.calibrations = calibrations

    return QasmQobj(qobj_id=qobj_id,
                    config=qobj_config,
                    experiments=experiments,
                    header=qobj_header)
