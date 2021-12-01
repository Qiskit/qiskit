# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Container class for backend options."""

import io


class Options:
    """Base options object

    This class is the abstract class that all backend options are based
    on. The properties of the class are intended to be all dynamically
    adjustable so that a user can reconfigure the backend on demand. If a
    property is immutable to the user (eg something like number of qubits)
    that should be a configuration of the backend class itself instead of the
    options.
    """

    _fields = {}

    def __init__(self, **kwargs):
        self._fields = {}
        self._fields.update(kwargs)
        self.validator = {}

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self._fields.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, Options) and isinstance(other, Options):
            return self._fields == other._fields
        return NotImplemented

    def set_validator(self, field, validator_value):
        """Set an optional validator for a field in the options

        Setting a validator enables changes to an options values to be
        validated for correctness when :meth:`~qiskit.providers.Options.update_options`
        is called. For example if you have a numeric field like
        ``shots`` you can specify a bounds tuple that set an upper and lower
        bound on the value such as::

            options.set_validator("shots", (1, 4096))

        In this case whenever the ``"shots"`` option is updated by the user
        it will enforce that the value is >=1 and <=4096. A ``ValueError`` will
        be raised if it's outside those bounds. If a validator is already present
        for the specified field it will be silently overriden.

        Args:
            field (str): The field name to set the validator on
            validator_value (list or tuple or type): The value to use for the
                validator depending on the type indicates on how the value for
                a field is enforced. If a tuple is passed in it must have a
                length of two and will enforce the min and max value
                (inclusive) for an integer or float value option. If it's a
                list it will list the valid values for a field. If it's a
                ``type`` the validator will just enforce the value is of a
                certain type.
        Raises:
            KeyError: If field is not present in the options object
            ValueError: If the ``validator_value`` has an invalid value for a
                given type
            TypeError: If ``validator_value`` is not a valid type
        """

        if field not in self._fields:
            raise KeyError("Field '%s' is not present in this options object" % field)
        if isinstance(validator_value, tuple):
            if len(validator_value) != 2:
                raise ValueError(
                    "A tuple validator must be of the form '(lower, upper)' "
                    "where lower and upper are the lower and upper bounds "
                    "inclusive of the numeric value"
                )
        elif isinstance(validator_value, list):
            if len(validator_value) == 0:
                raise ValueError("A list validator must have at least one entry")
        elif isinstance(validator_value, type):
            pass
        else:
            raise TypeError(
                f"{type(validator_value)} is not a valid validator type, it "
                "must be a tuple, list, or class/type"
            )
        self.validator[field] = validator_value

    def update_options(self, **fields):
        """Update options with kwargs"""
        for field in fields:
            field_validator = self.validator.get(field, None)
            if isinstance(field_validator, tuple):
                if fields[field] > field_validator[1] or fields[field] < field_validator[0]:
                    raise ValueError(
                        f"Specified value for '{field}' is not a valid value, "
                        f"must be >={field_validator[0]} or <={field_validator[1]}"
                    )
            elif isinstance(field_validator, list):
                if fields[field] not in field_validator:
                    raise ValueError(
                        f"Specified value for {field} is not a valid choice, "
                        f"must be one of {field_validator}"
                    )
            elif isinstance(field_validator, type):
                if not isinstance(fields[field], field_validator):
                    raise TypeError(
                        f"Specified value for {field} is not of required type {field_validator}"
                    )

        self._fields.update(fields)

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    def get(self, field, default=None):
        """Get an option value for a given key."""
        return getattr(self, field, default)

    def __str__(self):
        no_validator = super().__str__()
        if not self.validator:
            return no_validator
        else:
            out_str = io.StringIO()
            out_str.write(no_validator)
            out_str.write("\nWhere:\n")
            for field, value in self.validator.items():
                if isinstance(value, tuple):
                    out_str.write(f"\t{field} is >= {value[0]} and <= {value[1]}\n")
                elif isinstance(value, list):
                    out_str.write(f"\t{field} is one of {value}\n")
                elif isinstance(value, type):
                    out_str.write(f"\t{field} is of type {value}\n")
            return out_str.getvalue()
