# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Schema for backend status."""

from marshmallow.fields import String, Boolean, Integer
from marshmallow.validate import Range, Regexp

from .base import BaseModel, BaseSchema


class BackendStatusSchema(BaseSchema):
    """Schema for BackendStatus."""

    # Required fields.
    backend_name = String(required=True)
    backend_version = String(required=True,
                             validate=Regexp('[0-9]+.[0-9]+.[0-9]+$'))
    operational = Boolean(required=True)
    pending_jobs = Integer(required=True,
                           validate=Range(min=0))
    status_msg = String(required=True)


class BackendStatus(BaseModel):
    """Backend Status."""

    schema_cls = BackendStatusSchema

    def __init__(self, backend_name, backend_version, operational,
                 pending_jobs, status_msg,
                 **kwargs):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.operational = operational
        self.pending_jobs = pending_jobs
        self.status_msg = status_msg

        kwargs.update(self.__dict__)
        super().__init__(**kwargs)
