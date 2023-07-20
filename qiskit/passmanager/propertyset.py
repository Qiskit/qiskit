# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A property set dictionary that shared among optimization passes."""


from .exceptions import PassManagerError


class PropertySet(dict):
    """A default dictionary-like object."""

    def __missing__(self, key):
        return None


class FencedPropertySet(PropertySet):
    """A readonly property set that cannot be written via __setitem__."""

    def __setitem__(self, key, value):
        raise PassManagerError(
            "The fenced PropertySet has the property __setitem__ protected."
        )
