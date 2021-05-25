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

"""Experiment utility functions."""

import logging
from typing import Callable, Tuple, Dict, Any, Union
from functools import wraps
from datetime import datetime, timezone
import threading
from collections import OrderedDict
from abc import ABC, abstractmethod
import traceback
import pkg_resources

import dateutil.parser
from dateutil import tz

from qiskit.version import __version__ as terra_version

from .exceptions import ExperimentEntryNotFound, ExperimentEntryExists, ExperimentError

LOG = logging.getLogger(__name__)


def qiskit_version():
    """Return the Qiskit version."""
    try:
        return pkg_resources.get_distribution("qiskit").version
    except Exception:  # pylint: disable=broad-except
        return {"qiskit-terra": terra_version}


def parse_timestamp(utc_dt: Union[datetime, str]) -> datetime:
    """Parse a UTC ``datetime`` object or string.

    Args:
        utc_dt: Input UTC `datetime` or string.

    Returns:
        A ``datetime`` with the UTC timezone.

    Raises:
        TypeError: If the input parameter value is not valid.
    """
    if isinstance(utc_dt, str):
        utc_dt = dateutil.parser.parse(utc_dt)
    if not isinstance(utc_dt, datetime):
        raise TypeError("Input `utc_dt` is not string or datetime.")
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt


def utc_to_local(utc_dt: datetime) -> datetime:
    """Convert input UTC timestamp to local timezone.

    Args:
        utc_dt: Input UTC timestamp.

    Returns:
        A ``datetime`` with the local timezone.
    """
    local_dt = utc_dt.astimezone(tz.tzlocal())
    return local_dt


def save_data(
    is_new: bool, new_func: Callable, update_func: Callable, new_data: Dict, update_data: Dict
) -> Tuple[bool, Any]:
    """Save data in the database.

    Args:
        is_new: ``True`` if `new_func` should be called. Otherwise `update_func` is called.
        new_func: Function to create new entry in the database.
        update_func: Function to update an existing entry in the database.
        new_data: In addition to `update_data`, this data will be stored if creating
            a new entry.
        update_data: Data to be stored if updating an existing entry.

    Returns:
        A tuple of whether the data was saved and the function return value.

    Raises:
        ExperimentError: If unable to determine whether the entry exists.
    """
    attempts = 0
    try:
        # Attempt 3x for the unlikely scenario wherein is_new=False but the
        # entry doesn't actually exists. The second try might also fail if an entry
        # with the same ID somehow got created in the meantime.
        while attempts < 3:
            attempts += 1
            if is_new:
                try:
                    return True, new_func(**{**new_data, **update_data})
                except ExperimentEntryExists:
                    is_new = False
            else:
                try:
                    return True, update_func(**update_data)
                except ExperimentEntryNotFound:
                    is_new = True
        raise ExperimentError("Unable to determine the existence of the entry.")
    except Exception:  # pylint: disable=broad-except
        # Don't fail the experiment just because its data cannot be saved.
        LOG.error("Unable to save the experiment data: %s", traceback.format_exc())
        return False, None


def decorate_func(func: Callable, callback: Callable):
    """Decorate the input function."""

    @wraps(func)
    def _wrapped(*args, **kwargs):
        return_val = func(*args, **kwargs)
        callback()
        return return_val

    return _wrapped


class ThreadSafeContainer(ABC):
    """Base class for thread safe container."""

    def __init__(self, init_values=None):
        """ThreadSafeContainer constructor."""
        self._lock = threading.RLock()
        self._container = self._init_container(init_values)

    @abstractmethod
    def _init_container(self, init_values):
        """Initialize the container."""
        pass

    def __getitem__(self, key):
        with self._lock:
            return self._container[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._container[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._container[key]

    def __contains__(self, item):
        with self._lock:
            return item in self._container

    def __len__(self):
        with self._lock:
            return len(self._container)

    @property
    def lock(self):
        """Return lock used for this container."""
        return self._lock


class ThreadSafeOrderedDict(ThreadSafeContainer):
    """Thread safe OrderedDict."""

    def _init_container(self, init_values):
        """Initialize the container."""
        return OrderedDict.fromkeys(init_values or [])

    def get(self, key, default):
        """Return the value of the given key."""
        with self._lock:
            return self._container.get(key, default)

    def keys(self):
        """Return all key values."""
        with self._lock:
            return list(self._container.keys())

    def values(self):
        """Return all values."""
        with self._lock:
            return list(self._container.values())

    def items(self):
        """Return the key value pairs."""
        return self._container.items()


class ThreadSafeList(ThreadSafeContainer):
    """Thread safe list."""

    def _init_container(self, init_values):
        """Initialize the container."""
        return init_values or []

    def append(self, value):
        """Append to the list."""
        with self._lock:
            self._container.append(value)

    def copy(self):
        """Returns a copy of the list."""
        with self.lock:
            return self._container.copy()
