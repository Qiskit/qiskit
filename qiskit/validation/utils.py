# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Collection of utilities, helpers and special values to use with
marshmallow."""

from traceback import StackSummary, walk_stack


class _ValidModel:
    """Represents a valid model."""

    def __bool__(self):
        return True

    def __repr__(self):
        return '<qiskit.validation.valid_model>'


VALID_MODEL = _ValidModel()


def is_validating():
    """Check if execution is in the middle of a model validation.

    This function _climbs_ the stacktrace in the search of a local flag
    indicating the execution is in the middle of a model validation.
    """
    this_traceback = walk_stack(None)
    frames = StackSummary.extract(this_traceback, capture_locals=True)
    for frame in frames:
        if frame.locals.get('__is_validating__', False):
            return True

    return False
