# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Schema-conformant representation of Results."""

from qiskit.qobj import QobjItem


class Result(QobjItem):
    """Schema-conformant Result.

    Attributes:
        backend_name (string): Backend name.
        backend_version (string): Backend version in the form X.X.X.
            If there is only one shot to the experiment this is [1,1]
        qobj_id (string): User generated Qobj id.
        job_id (string): Unique execution id from the backend.
        success (boolean): True if complete input qobj executed correctly.
            (Implies each experiment success)
        results (list[ExperimentResult]): Corresponding results for array of
            experiments of the input qobj

    Attributes defined in the schema but not required:
        status (string): Human-readable status of complete qobj execution.
        date (string): Date/time of job execution.
        header (dict): Header passed through from the qobj with job metadata.
    """

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
        success (bool): If true, we can trust results for this experiment.
        shots (int): The starting and ending shot for this data.
            If there is only one shot to the experiment this is [1,1]
        data (dict): result data for the experiment.

    Attributes defined in the schema but not required:
        status (string): Human-readable description of status of this
        experiment.
        seed (string): Experiment-level random seed.
        mass_return (string): Is the data in the memory/snapshot averaged or
        from each shot.
        header (dict): Header passed through from the qobj with experiment
        metadata
    """

    REQUIRED_ARGS = ['success', 'shots', 'data']

    def __init__(self, success, shots, data, **kwargs):
        self.success = success
        self.shots = shots
        self.data = data

        super().__init__(**kwargs)
