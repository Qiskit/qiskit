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

"""Class for job status."""

from types import SimpleNamespace


class JobStatus(SimpleNamespace):
    """Model for JobStatus.

    Attributes:
        job_id (str): backend job_id.
        status (str): status of the job.
        status_msg (str): status message.
    """

    def __init__(self, job_id, status, status_msg, **kwargs):
        self.job_id = job_id
        self.status = status
        self.status_msg = status_msg
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data):
        """Create a new JobStatus object from a dictionary.

        Args:
            data (dict): A dictionary representing the JobStatus to create.
                         It will be in the same format as output by
                         :meth:`to_dict`.

        Returns:
            qiskit.providers.model.JobStatus: The ``JobStatus`` from the input
                dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the JobStatus.

        Returns:
            dict: The dictionary form of the JobStatus.
        """
        return self.__dict__

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        return self.from_dict(state)

    def __reduce__(self):
        return (self.__class__, (self.job_id, self.status, self.status_msg))
