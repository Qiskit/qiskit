# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Compile function for converting a list of circuits to the qobj"""
import uuid
import warnings

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjInstruction, QobjHeader
from qiskit.qobj import QobjExperimentConfig, QobjExperimentHeader, QobjConditional
from qiskit.qobj.run_config import RunConfig


def circuits_to_qobj(circuits, user_qobj_header=None, run_config=None,
                     qobj_id=None, backend_name=None,
                     config=None, shots=None, max_credits=None,
                     basis_gates=None,
                     coupling_map=None, seed=None, memory=None):
    """Convert a list of circuits into a qobj.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit): circuits to compile
        user_qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): RunConfig object
        qobj_id (int): identifier for the generated qobj

        backend_name (str): TODO: delete after qiskit-terra 0.8
        config (dict): TODO: delete after qiskit-terra 0.8
        shots (int): TODO: delete after qiskit-terra 0.8
        max_credits (int): TODO: delete after qiskit-terra 0.8
        basis_gates (str): TODO: delete after qiskit-terra 0.8
        coupling_map (list): TODO: delete after qiskit-terra 0.8
        seed (int): TODO: delete after qiskit-terra 0.8
        memory (bool): TODO: delete after qiskit-terra 0.8

    Returns:
        Qobj: the Qobj to be run on the backends
    """
    user_qobj_header = user_qobj_header or QobjHeader()
    run_config = run_config or RunConfig()
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    if backend_name:
        warnings.warn('backend_name is not required anymore', DeprecationWarning)
        user_qobj_header.backend_name = backend_name
    if config:
        warnings.warn('config is not used anymore. Set all configs in '
                      'run_config.', DeprecationWarning)
    if shots:
        warnings.warn('shots is not used anymore. Set it via run_config.', DeprecationWarning)
        run_config.shots = shots
    if basis_gates:
        warnings.warn('basis_gates was unused and will be removed.', DeprecationWarning)
    if coupling_map:
        warnings.warn('coupling_map was unused and will be removed.', DeprecationWarning)
    if seed:
        warnings.warn('seed is not used anymore. Set it via run_config', DeprecationWarning)
        run_config.seed = seed
    if memory:
        warnings.warn('memory is not used anymore. Set it via run_config', DeprecationWarning)
        run_config.memory = memory
    if max_credits:
        warnings.warn('max_credits is not used anymore. Set it via run_config', DeprecationWarning)
        run_config.max_credits = max_credits

    userconfig = QobjConfig(**run_config.to_dict())
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
        # TODO: when no more backends use the compiled_circuit_qasm lets delete it form header
        experimentheader = QobjExperimentHeader(qubit_labels=qubit_labels,
                                                n_qubits=n_qubits,
                                                qreg_sizes=qreg_sizes,
                                                clbit_labels=clbit_labels,
                                                memory_slots=memory_slots,
                                                creg_sizes=creg_sizes,
                                                name=circuit.name,
                                                compiled_circuit_qasm=circuit.qasm())
        # TODO: why do we need n_qubits and memory_slots in both the header and the config
        experimentconfig = QobjExperimentConfig(n_qubits=n_qubits, memory_slots=memory_slots)

        instructions = []
        for opt in circuit.data:
            current_instruction = QobjInstruction(name=opt.name)
            if opt.qargs:
                qubit_indices = [qubit_labels.index([qubit[0].name, qubit[1]])
                                 for qubit in opt.qargs]
                current_instruction.qubits = qubit_indices
            if opt.cargs:
                clbit_indices = [clbit_labels.index([clbit[0].name, clbit[1]])
                                 for clbit in opt.cargs]
                current_instruction.memory = clbit_indices
            # TODO: we are not constant with params vs param
            if opt.param:
                params = list(map(lambda x: x.evalf(), opt.param))
                current_instruction.params = params

            # TODO: I really dont like this for snapshot. I also think we should change
            # type to snap_type
            if opt.name == "snapshot":
                current_instruction.label = str(opt.param[0])
                current_instruction.type = str(opt.param[1])
            if opt.control:
                mask = 0
                for clbit in clbit_labels:
                    if clbit[0] == opt.control[0].name:
                        mask |= (1 << clbit_labels.index(clbit))

                current_instruction.conditional = QobjConditional(mask="0x%X" % mask,
                                                                  type='equals',
                                                                  val="0x%X" % opt.control[1])

            instructions.append(current_instruction)
        experiments.append(QobjExperiment(instructions=instructions, header=experimentheader,
                                          config=experimentconfig))
        if n_qubits > max_n_qubits:
            max_n_qubits = n_qubits
        if memory_slots > max_memory_slots:
            max_memory_slots = memory_slots

    userconfig.memory_slots = max_memory_slots
    userconfig.n_qubits = max_n_qubits

    return Qobj(qobj_id=qobj_id or str(uuid.uuid4()), config=userconfig,
                experiments=experiments, header=user_qobj_header)
