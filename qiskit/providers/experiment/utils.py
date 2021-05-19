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

import logging
from typing import Callable, Optional, Tuple, Dict, Any, Union, List
from functools import wraps
import pkg_resources
from datetime import datetime, timezone
import threading
from collections import OrderedDict

import dateutil.parser
from dateutil import tz

from qiskit.version import __version__ as terra_version

from .exceptions import ExperimentEntryNotFound, ExperimentEntryExists, ExperimentError

LOG = logging.getLogger(__name__)


def qiskit_version():
    """Return the Qiskit version."""
    try:
        return pkg_resources.get_distribution('qiskit').version
    except Exception:  # pylint: disable=broad-except
        return {'qiskit-terra': terra_version}


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
        raise TypeError('Input `utc_dt` is not string or datetime.')
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
        is_new: bool,
        new_func: Callable,
        update_func: Callable,
        new_data: Dict,
        update_data: Dict
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
    except Exception as ex:
        # Don't fail the experiment just because its data cannot be saved.
        LOG.error(f"Unable to save the experiment data: {str(ex)}")
        return False, None


def decorate_func(func: Callable, callback: Callable):
    """Decorate the input function."""
    @wraps(func)
    def _wrapped(*args, **kwargs):
        return_val = func(*args, **kwargs)
        callback()
        return return_val
    return _wrapped


class MonitoredList(list):
    """A list class with a callback function.

    Use :meth:`create_with_callback` method to create an instance of
    this class. The callback function is invoked when the data inside the
    list changes.
    """

    @classmethod
    def create_with_callback(
            cls,
            callback: Callable,
            init_data: Optional[list] = None
    ) -> 'MonitoredList':
        """Create an instance with a callback function.

        Args:
            callback: The callback function to invoke when data inside this
                list changes.
            init_data: Initial data used to populate the list.

        Returns:
            An instance of this class.
        """
        obj = cls()
        if init_data:
            obj.append(init_data)

        monitored = ['__setitem__', '__delitem__', '__add__', 'clear', 'append',
                     'insert', 'extend', 'pop', 'remove']
        for key, value in list.__dict__.items():
            if key in monitored:
                setattr(cls, key, decorate_func(value, callback))

        return obj


class MonitoredDict(dict):
    """A dictionary class with a callback function.

    Use :meth:`create_with_callback` method to create an instance of
    this class. The callback function is invoked when the data inside the
    dictionary changes.
    """

    @classmethod
    def create_with_callback(
            cls,
            callback: Callable,
            init_data: Optional[dict] = None
    ) -> 'MonitoredDict':
        """Create an instance with a callback function.

        Args:
            callback: The callback function to invoke when data inside this
                dictionary changes.
            init_data: Initial data used to populate the dictionary.

        Returns:
            An instance of this class.
        """
        obj = cls()
        if init_data:
            obj.update(init_data)

        monitored = ['__setitem__', '__delitem__', 'setdefault', 'pop', 'popitem',
                     'update', 'clear']
        for key, value in dict.__dict__.items():
            if key in monitored:
                setattr(cls, key, decorate_func(value, callback))
        return obj


class ThreadSafeOrderedDict:

    def __init__(self, init_keys: Optional[List] = None):
        self._lock = threading.Lock()
        init_keys = init_keys or []
        self._dict = OrderedDict.fromkeys(init_keys)

    def set_item(self, key: Any, val: Any):
        with self._lock:
            self._dict[key] = val

    def del_item(self, key: Any):
        with self._lock:
            del self._dict[key]

    def item(self, key: Any) -> Any:
        with self._lock:
            return self._dict[key]

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self) -> List:
        with self._lock:
            return list(self._dict.values())
