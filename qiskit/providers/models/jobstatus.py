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

"""Model and schema for job status."""

from marshmallow.validate import OneOf

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import String


class JobStatusSchema(BaseSchema):
    """Schema for JobStatus."""

    # Required properties.
    job_id = String(required=True)
    status = String(required=True,
                    validate=OneOf(['DONE', 'QUEUED', 'CANCELLED', 'RUNNING', 'ERROR']))
    status_msg = String(required=True)


@bind_schema(JobStatusSchema)
class JobStatus(BaseModel):
    """Model for JobStatus.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``JobStatusSchema``.

    Attributes:
        job_id (str): backend job_id.
        status (str): status of the job.
        status_msg (str): status message.
    """

    def __init__(self, job_id, status, status_msg, **kwargs):
        self.job_id = job_id
        self.status = status
        self.status_msg = status_msg

        super().__init__(**kwargs)
