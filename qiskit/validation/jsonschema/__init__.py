# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Validation using the .json schemas."""

from .exceptions import SchemaValidationError
from .schema_validation import validate_json_against_schema
