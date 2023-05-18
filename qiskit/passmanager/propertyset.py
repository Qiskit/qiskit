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

"""A property set is maintained by the pass runner.


This is sort of shared memory space among passes.
"""

from contextvars import ContextVar
from .exceptions import PassManagerError


CONTEXT_PROPERTIES = ContextVar("property_set")


class PropertySet(dict):
    """A default dictionary-like object."""

    def __missing__(self, key):
        return None


class FuturePropertySet(PropertySet):
    """A handler of the property set, which might not be created yet.

    This instance can be initialized outside the thread that a pass runner is running.
    In other words, it can be created prior to initialization of the pass runner.
    When attribute of this property set is accessed, it takes a value from
    the thread local property set through the Python ContextVar.

    This object is usually attached to the flow controllers to evaluate its iterator condition,
    and thus this handler only provides read-only access to the context property set.
    """

    def __getitem__(self, item):
        return get_property_set().get(item, None)

    def __setitem__(self, key, value):
        raise PassManagerError(
            f"The fenced {type(PropertySet)} has the property __setitem__ protected."
        )


def get_property_set() -> PropertySet:
    """A helper function to get property set from the thread local storage.

    Returns:
        A property set created in the currently running thread.
        If returned property set is modified, it mutates the original object
        in the thread local storage.

    Raises:
        PassManagerError: When this property set is not found in current thread.
    """
    try:
        return CONTEXT_PROPERTIES.get()
    except LookupError as ex:
        raise PassManagerError(
            "Property set is not found. "
            "A property set might be called outside of the thread the pass runner is running, "
            "or property set might not be initialized in this thread."
        ) from ex


def init_property_set(**kwargs):
    """A helper function to initialize property set in current thread.

    Args:
        kwargs: Arbitrary initial status of the pass manager.
    """
    property_set = PropertySet(**kwargs)
    CONTEXT_PROPERTIES.set(property_set)
