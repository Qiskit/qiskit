# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Qobj conversion helpers."""
import logging

from qiskit import QiskitError
from ._qobj import QOBJ_VERSION
from ._qobj import QobjItem


logger = logging.getLogger(__name__)


def qobj_to_dict(qobj, version=QOBJ_VERSION):
    """Convert a Qobj to another version of the schema.
    Convert all types to native python types.

    Args:
        qobj (Qobj): input Qobj.
        version (string): target version for conversion.

    Returns:
        dict: dictionary representing the qobj for the specified schema version.

    Raises:
        QiskitError: if the target version is not supported.
    """
    if version == QOBJ_VERSION:
        return qobj_to_dict_current_version(qobj)
    elif version == '0.0.1':
        return_dict = qobj_to_dict_previous_version(qobj)
        return {key: QobjItem._expand_item(value) for key, value
                in return_dict.items()}
    else:
        raise QiskitError('Invalid target version for conversion.')


def qobj_to_dict_current_version(qobj):
    """
    Return a dictionary representation of the QobjItem, recursively converting
    its public attributes.

    Args:
        qobj (Qobj): input Qobj.

    Returns:
        dict: dictionary representing the qobj.
    """
    return qobj.as_dict()


def qobj_to_dict_previous_version(qobj):
    """Convert a Qobj to the 0.0.1 version of the schema.

    Args:
        qobj (Qobj): input Qobj.

    Returns:
        dict: dictionary representing the qobj for the specified schema version.
    """
    # Build the top Qobj element.
    converted = {
        'id': qobj.qobj_id,
        'config': {
            'shots': qobj.config.shots,
            'backend_name': getattr(qobj.header, 'backend_name', None),
            'max_credits': getattr(qobj.config, 'max_credits', None)
        },
        'circuits': []
    }

    # Update configuration: qobj.config might have extra items.
    for key, value in qobj.config.__dict__.items():
        if key not in ('shots', 'memory_slots', 'max_credits', 'seed'):
            converted['config'][key] = value

    # Add circuits.
    for experiment in qobj.experiments:
        circuit_config = getattr(experiment, 'config', {})
        if circuit_config:
            circuit_config = circuit_config.as_dict()
            circuit_config['seed'] = getattr(qobj.config, 'seed', None)

        circuit = {
            'name': getattr(experiment.header, 'name', None),
            'config': circuit_config,
            'compiled_circuit': {
                'header': experiment.header.as_dict(),
                'operations': [instruction.as_dict() for instruction in
                               experiment.instructions]
            },
            'compiled_circuit_qasm': getattr(experiment.header,
                                             'compiled_circuit_qasm', None)
        }

        converted['circuits'].append(circuit)

    return converted
