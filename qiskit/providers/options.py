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

from collections.abc import MutableMapping


class Options(MutableMapping):
    """Base options object

    This class is the abstract class that all backend options are based
    on. The properties of the class are intended to be all dynamically
    adjustable so that a user can reconfigure the backend on demand. If a
    property is immutable to the user (eg something like number of qubits)
    that should be a configuration of the backend class itself instead of the
    options.
    """

    _dict = {}

    def __init__(self, **kwargs):
        self._dict = {}
        super().__init__()
        self._dict = kwargs

    def __getattr__(self, field):
        if field == "_dict":
            super().__getattribute__(field)
        elif field not in self._dict:
            raise AttributeError("%s not a valid option" % field)
        return self._dict[field]

    def __getitem__(self, field):
        if field not in self._dict:
            raise KeyError("%s not a valid option" % field)
        self._dict.get(field)

    def __setitem__(self, field, value):
        if field in self._dict:
            self._dict[field] = value
        else:
            raise KeyError("%s not a valid option" % field)

    def __delitem__(self, field):
        del self._dict[field]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __setattr__(self, field, value):
        if field == "_dict":
            super().__setattr__(field, value)
        elif field in self._dict:
            self._dict[field] = value
        else:
            raise AttributeError("%s not a valid option" % field)

    def __repr__(self):
        return "Options(%s)" % ", ".join("k=v" for k, v in self._dict.items())

    def update_options(self, **fields):
        """Update options with kwargs"""
        self._dict.update(fields)

    def get(self, field, default=None):  # pylint: disable=arguments-differ
        """Get an option value for a given key."""
        return self._dict.get(field, default)

    def __dict__(self):
        return self._dict

    def __eq__(self, other):
        if isinstance(other, Options):
            return self._dict == other._dict
        return NotImplementedError
