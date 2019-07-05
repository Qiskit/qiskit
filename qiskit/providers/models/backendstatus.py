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

"""Model and schema for backend status."""

from marshmallow.validate import Range, Regexp

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Boolean, Integer, String


class BackendStatusSchema(BaseSchema):
    """Schema for BackendStatus."""

    # Required properties.
    backend_name = String(required=True)
    backend_version = String(required=True,
                             validate=Regexp("[0-9]+.[0-9]+.[0-9]+$"))
    operational = Boolean(required=True)
    pending_jobs = Integer(required=True,
                           validate=Range(min=0))
    status_msg = String(required=True)


@bind_schema(BackendStatusSchema)
class BackendStatus(BaseModel):
    """Model for BackendStatus.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``BackendStatusSchema``.

    Attributes:
        backend_name (str): backend name.
        backend_version (str): backend version in the form X.Y.Z.
        operational (bool): backend operational and accepting jobs.
        pending_jobs (int): number of pending jobs on the backend.
        status_msg (str): status message.
    """

    def __init__(self, backend_name, backend_version, operational,
                 pending_jobs, status_msg, **kwargs):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.operational = operational
        self.pending_jobs = pending_jobs
        self.status_msg = status_msg

        super().__init__(**kwargs)
