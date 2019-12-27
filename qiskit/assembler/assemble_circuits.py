# -*- coding: utf-8 -*-

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
from qiskit.qobj import (QasmQobj, QobjExperimentHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment,
                         QasmQobjConfig)


def assemble_circuits(circuits, run_config, qobj_id, qobj_header):
    """Assembles a list of circuits into a qobj that can be run on the backend.

    Args:
        circuits (list[QuantumCircuit]): circuit(s) to assemble
        qobj_id (int): identifier for the generated qobj
        qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): configuration of the runtime environment

    Returns:
        QasmQobj: the qobj to be run on the backends
    """
    qobj_config = QasmQobjConfig()
    if run_config:
        qobj_config = QasmQobjConfig(**run_config.to_dict())

    # Pack everything into the Qobj
    experiments = []
    max_n_qubits = 0
    max_memory_slots = 0
    for circuit in circuits:
        # header stuff
        n_qubits = 0
        memory_slots = 0
        qubit_labels = []
        clbit_labels = []

        qreg_sizes = []
        creg_sizes = []
        for qreg in circuit.qregs:
            qreg_sizes.append([qreg.name, qreg.size])
            for j in range(qreg.size):
                qubit_labels.append([qreg.name, j])
            n_qubits += qreg.size
        for creg in circuit.cregs:
            creg_sizes.append([creg.name, creg.size])
            for j in range(creg.size):
                clbit_labels.append([creg.name, j])
            memory_slots += creg.size

        # TODO: why do we need creq_sizes and qreg_sizes in header
        # TODO: we need to rethink memory_slots as they are tied to classical bit
        header = QobjExperimentHeader(qubit_labels=qubit_labels,
                                      n_qubits=n_qubits,
                                      qreg_sizes=qreg_sizes,
                                      clbit_labels=clbit_labels,
                                      memory_slots=memory_slots,
                                      creg_sizes=creg_sizes,
                                      name=circuit.name)
        # TODO: why do we need n_qubits and memory_slots in both the header and the config
        config = QasmQobjExperimentConfig(n_qubits=n_qubits, memory_slots=memory_slots)

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
                                                       register=conditional_reg_idx,
                                                       validate=False)
                instructions.append(conversion_bfunc)
                instruction.conditional = conditional_reg_idx
                max_conditional_idx += 1
                # Delete condition attribute now that we have replaced it with
                # the conditional and bfuc
                del instruction._condition

            instructions.append(instruction)

        experiments.append(QasmQobjExperiment(instructions=instructions, header=header,
                                              config=config))
        if n_qubits > max_n_qubits:
            max_n_qubits = n_qubits
        if memory_slots > max_memory_slots:
            max_memory_slots = memory_slots

    qobj_config.memory_slots = max_memory_slots
    qobj_config.n_qubits = max_n_qubits

    return QasmQobj(qobj_id=qobj_id,
                    config=qobj_config,
                    experiments=experiments,
                    header=qobj_header)
