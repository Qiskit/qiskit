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

"""Qobj utilities and enums."""

from enum import Enum, IntEnum
import warnings

from fastjsonschema.exceptions import JsonSchemaException

from qiskit.validation.jsonschema.exceptions import SchemaValidationError


class QobjType(str, Enum):
    """Qobj.type allowed values."""

    QASM = "QASM"
    PULSE = "PULSE"


class MeasReturnType(str, Enum):
    """PulseQobjConfig meas_return allowed values."""

    AVERAGE = "avg"
    SINGLE = "single"


class MeasLevel(IntEnum):
    """MeasLevel allowed values."""

    RAW = 0
    KERNELED = 1
    CLASSIFIED = 2


def validate_qobj_against_schema(qobj):
    """Validates a QObj against the .json schema.

    Args:
        qobj (Qobj): Qobj to be validated.

    Raises:
        SchemaValidationError: if the qobj fails schema validation
    """
    warnings.warn(
        "The jsonschema validation included in qiskit-terra is "
        "deprecated and will be removed in a future release. "
        "If you're relying on this schema validation you should "
        "pull the schemas from the Qiskit/ibmq-schemas and directly "
        "validate your payloads with that",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        qobj.to_dict(validate=True)
    except JsonSchemaException as err:
        raise SchemaValidationError(
            f"Qobj validation failed. Specifically path: {err.path}"  # pylint: disable=no-member
            f" failed to fulfil {err.definition}"
        ) from err
