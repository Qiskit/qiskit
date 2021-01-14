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
from typing import Callable, Optional, Tuple, Dict, Any
from functools import wraps

from .exceptions import ExperimentDataNotFound, ExperimentDataExists, ExperimentError

LOG = logging.getLogger(__name__)


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
                except ExperimentDataExists:
                    is_new = False
            else:
                try:
                    return True, update_func(**update_data)
                except ExperimentDataNotFound:
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

    @classmethod
    def create_with_callback(cls, callback: Callable, init_data: Optional[list] = None):
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

    @classmethod
    def create_with_callback(cls, callback: Callable, init_data: Optional[dict] = None):
        obj = cls()
        if init_data:
            obj.update(init_data)

        monitored = ['__setitem__', '__delitem__', 'setdefault', 'pop', 'popitem',
                     'update', 'clear']
        for key, value in dict.__dict__.items():
            if key in monitored:
                setattr(cls, key, decorate_func(value, callback))
        return obj
