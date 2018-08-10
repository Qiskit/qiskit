# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Schema-conformant representation of Results."""

from qiskit.qobj import QobjItem


class Result(QobjItem):
    """Schema-conformant Result."""

    REQUIRED_ARGS = ['backend_name', 'backend_version', 'qobj_id', 'job_id',
                     'success', 'results']

    def __init__(self, backend_name, backend_version, qobj_id, job_id,
                 success, results, **kwargs):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.qobj_id = qobj_id
        self.job_id = job_id
        self.success = success
        self.results = results

        super().__init__(**kwargs)


class ExperimentResult(QobjItem):
    """Schema-conformant Experiment result.

    Attributes:
        success (bool):
        shots (int): number of shots.
        data (dict):
    """

    REQUIRED_ARGS = ['success', 'shots', 'data']

    def __init__(self, success, shots, data, **kwargs):
        self.success = success
        self.shots = shots
        self.data = data

        super().__init__(**kwargs)
