# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
