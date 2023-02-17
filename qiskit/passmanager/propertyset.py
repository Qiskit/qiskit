# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" A property set is maintained by the PassManager to keep information
about the current state of the circuit """

from contextvars import ContextVar

from .exceptions import PassManagerError

CONTEXT_PROPERTIES = ContextVar("property_set")


class PropertySet(dict):
    """A default dictionary-like object"""

    def __missing__(self, key):
        return None


class FuturePropertySet(PropertySet):
    """A property set placeholder.

    This instance can be initialized outside the target thread.
    Any attribute of the property set is accessed,
    it takes value from the thread local property set through the contextvar.

    This provides read-only access to the context property set.
    """

    def __getitem__(self, key):
        return get_property_set().get(key, None)

    def __setitem__(self, key, value):
        raise PassManagerError(
            f"The fenced {type(PropertySet)} has the property __setitem__ protected."
        )


def get_property_set() -> PropertySet:
    """A helper function to get property set from the thread local storage.

    Returns:
        A thread local property set.

    Raises:
        PassManagerError: When this function is not called outside the thread.
    """
    try:
        return CONTEXT_PROPERTIES.get()
    except LookupError as ex:
        raise PassManagerError(
            "A property set is called outside the running thread or "
            "property set is not initialized in this thread."
        ) from ex


def init_property_set(**kwargs):
    """A helper function to initialize property set in the thread.

    Args:
        kwargs: Arbitrary initial status of the property set.
    """
    property_set = PropertySet(**kwargs)
    CONTEXT_PROPERTIES.set(property_set)
