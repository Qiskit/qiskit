# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Utilities for working with results."""

import logging

from . import Result


logger = logging.getLogger(__name__)


def result_from_old_style_dict(result_dict):
    """Return a `Result` from a dict that is using the previous format.

    Args:
        result_dict (dict): dictionary in the old format.
    Returns:
        qiskit.Result: a Result instance.
    """
    # Prepare the experiment results: "counts" keys as hex.
    for experiment_result in result_dict['result']:
        if 'counts' in experiment_result['data']:
            counts = experiment_result['data']['counts']
            keys = list(counts.keys())
            for key in keys:
                key_as_hex = hex(int(key.replace(' ', ''), 2))
                counts[key_as_hex] = counts.pop(key)

    # TODO: simulators return `backend`, ibmq seems to return `backend_name`.
    # The schema expects `backend_name`.
    if 'backend' in result_dict:
        result_dict['backend_name'] = result_dict['backend']
        del result_dict['backend']

    # TODO: some fields are missing. This should be revised when everything
    # outputs schema-conformant results (including `test_ibmqjob_states`).
    result_dict.update({
        'backend_version': result_dict.get('backend_version', '0.0.0'),
        'job_id': result_dict.get('job_id', 'TODO'),
        'success': result_dict.get('success', False),
        'backend_name': result_dict.get('backend_name', 'TODO'),
        'qobj_id': result_dict.pop('id'),
        'results': result_dict.pop('result')
    })

    return Result.from_dict(result_dict)


def copy_qasm_from_qobj_into_result(qobj_, result):
    """Copy QASMs belonging to the Qobj experiment into a Result.

    Find the QASMs belonging to the Qobj experiments and copy them
    into the corresponding result entries.

    Args:
        qobj_ (qobj): Qobj
        result (qiskit.Result): Result (modified in-place).
    """
    for experiment in qobj_.experiments:
        name = experiment.header.name
        qasm = getattr(experiment.header, 'compiled_circuit_qasm', None)
        experiment_result = _find_experiment_result(result, name)
        if qasm and experiment_result:
            experiment_result['compiled_circuit_qasm'] = qasm

        # TODO: passing the header to the results should be done at a higher
        # level. This ensures result[x].header.name is present, for results.
        experiment_result['header'] = experiment.header.as_dict()


def _find_experiment_result(result, name):
    for experiment_result in result['results']:
        if experiment_result['name'] == name:
            return experiment_result

    logger.warning('No result found for experiment %s', name)
    return None
