# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Fields custom to Terra to be used with Qiskit validated classes."""

import numpy
import sympy

from marshmallow.utils import is_collection

from qiskit.validation import ModelTypeValidator


class Complex(ModelTypeValidator):
    """Field for complex numbers.

    Field for parsing complex numbers:
    * deserializes to Python's `complex`.
    * serializes to a tuple of 2 decimals `(real, imaginary)`
    """

    valid_types = (complex, )

    default_error_messages = {
        'invalid': '{input} cannot be parsed as a complex number.',
        'format': '"{input}" cannot be formatted as complex number.',
    }

    def _serialize(self, value, attr, obj):
        try:
            return [value.real, value.imag]
        except AttributeError:
            self.fail('format', input=value)

    def _deserialize(self, value, attr, data):
        if not is_collection(value) or len(value) != 2:
            self.fail('invalid', input=value)

        try:
            return complex(*value)
        except (ValueError, TypeError):
            self.fail('invalid', input=value)


class InstructionParameter(ModelTypeValidator):
    """Field for objects used in instruction parameters.

    This field provides support for parsing objects of types that uses by
    qobj.experiments.instructions.parameters:
    * basic Python types: complex, int, float, str
    * ``numpy``: integer, float
    * ``sympy``: Symbol, Basic

    Note that by using this field, serialization-deserialization round-tripping
    becomes not possible, as certain types serialize to the same Python basic
    type (for example, numpy.float and regular float). If possible, it is
    recommended that more specific and defined fields are used instead.
    """
    valid_types = (complex, int, float, str,
                   numpy.integer, numpy.float, sympy.Basic, sympy.Symbol)

    def _serialize(self, value, attr, obj):
        # pylint: disable=too-many-return-statements
        if isinstance(value, (float, int, str)):
            return value
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, numpy.integer):
            return int(value)
        if isinstance(value, numpy.float):
            return float(value)
        if isinstance(value, sympy.Symbol):
            return str(value)
        if isinstance(value, sympy.Basic):
            if value.is_imaginary:
                return [float(sympy.re(value)), float(sympy.im(value))]
            else:
                return float(value.evalf())

        return self.fail('invalid', input=value)

    def _deserialize(self, value, attr, data):
        if is_collection(value) and len(value) != 2:
            return complex(*value)

        return value
