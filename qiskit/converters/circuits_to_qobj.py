# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Compile function for converting a list of circuits to the qobj"""
import warnings

from qiskit.qobj import QobjHeader
from qiskit.compiler.run_config import RunConfig
from qiskit.compiler import assemble_circuits


def circuits_to_qobj(circuits, qobj_header=None, run_config=None,
                     qobj_id=None, backend_name=None,
                     config=None, shots=None, max_credits=None,
                     basis_gates=None,
                     coupling_map=None, seed=None, memory=None):
    """Convert a list of circuits into a qobj.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit): circuits to compile
        qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): RunConfig object

        qobj_id (int): TODO: delete after qiskit-terra 0.8
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
    warnings.warn('circuits_to_qobj is deprecated and will be removed in Qiskit Terra 0.9. '
              'Use qiskit.compiler.assemble() to serialize circuits into a qobj.',
              DeprecationWarning)

    qobj_header = qobj_header or QobjHeader()
    run_config = run_config or RunConfig()

    if backend_name:
        warnings.warn('backend_name is not required anymore', DeprecationWarning)
        qobj_header.backend_name = backend_name
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
    if qobj_id:
        warnings.warn('qobj_id is not used anymore', DeprecationWarning)

    qobj = assemble_circuits(circuits, qobj_header, run_config)

    return qobj
