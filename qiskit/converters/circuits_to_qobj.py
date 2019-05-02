# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compile function for converting a list of circuits to the qobj"""
import warnings

from qiskit.qobj import QobjHeader
from qiskit.compiler import assemble


def circuits_to_qobj(circuits, qobj_header=None,
                     qobj_id=None, backend_name=None,
                     config=None, shots=None, max_credits=None,
                     basis_gates=None,
                     coupling_map=None, seed=None, memory=None):
    """Convert a list of circuits into a qobj.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit): circuits to compile
        qobj_header (QobjHeader): header to pass to the results
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

    if backend_name:
        qobj_header.backend_name = backend_name
    if basis_gates:
        warnings.warn('basis_gates was unused and will be removed.', DeprecationWarning)
    if coupling_map:
        warnings.warn('coupling_map was unused and will be removed.', DeprecationWarning)

    qobj = assemble(experiments=circuits,
                    qobj_id=qobj_id,
                    qobj_header=qobj_header,
                    shots=shots,
                    memory=memory,
                    max_credits=max_credits,
                    seed_simulator=seed,
                    config=config)

    return qobj
