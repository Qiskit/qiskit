# -*- coding: utf-8 -*-

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

from abc import ABC
from abc import abstractmethod


class Backend(ABC):
    """Abstract class for Backends

    This abstract class is to be used for all Backend objects created by a
    provider. There are several classes of information contained in a Backend.
    The first are the properties of class itself. These should be used to
    defined the immutable characteristics of the backend. For a backend
    that runs experiments this would be things like ``basis_gates`` that
    do not change for the lifetime of the backend. The ``options``
    attribute of the backend is used to contain the dynamic properties of
    the backend. The intent is that these will all be user configurable. It
    should be used more for runtime properties that **configure** how the
    backend is used. For example, something like a ``shots`` field for a
    backend that runs experiments which would contain an int for how many
    shots to execute. The ``properties`` attribute is optionally defined and
    is used to return measured properties, or properties of a backend that may
    change over time. The simplest example of this would be a version string,
    which will change as a backend is updated, but also could be something like
    noise parameters for backends that run experiments.

    For Backends that will run circuits you'll want to have a target defined
    as well. This will provide necessary information to the transpiler so that
    circuits will be transpiled so that they actually run on the backend.
    """

    version = 1

    def __init__(self, name, **fields):
        """Initialize a backend class

        Args:
            name (str): The name of the backend
            fields: kwargs for the values to use to override the default options.
        Raises:
            AttributeError: if input field not a valid options
        """
        self.name = name
        self._options = self._default_config()
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError(
                        "Options field %s is not valid for this backend" % field)
            self._options.update_config(**fields)

    @classmethod
    @abstractmethod
    def _default_config(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.v2.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.v2.Options: A options object with
                default values set
        """
        pass

    def set_options(self, **fields):
        """Set the options fields for the backend

        This method is used to update the options of a backend. If
        you need to change any of the options prior to running just
        pass in the kwarg with the new value for the options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not part of the
                options
        """
        for field in fields:
            if field not in self._options.data:
                raise AttributeError(
                    "Options field %s is not valid for this "
                    "backend" % field)
        self._options.update_config(**fields)

    @property
    def properties(self):
        """Return the backend's measure properties.

        The properties of a backend represent measure properties of a backend.
        These are fields that are immutable for the property but may change
        over time based on the state of the Backend.

        This is an optional property for the Backend and will return None if a
        backend does not have properties.
        """
        if hasattr(self, '_properties'):
            return self._properties
        else:
            return None

    @property
    def options(self):
        """Return the options for the backend

        The options of a backend are the dynamic parameters defining
        how the backend is used. These are used to control the :meth:`run`
        method.
        """
        return self._options

    @property
    def target(self):
        """Return the backend's target information

        The target of a backend represents the information for the compiler
        to set this backend as the target device. These are fields that are
        immutable for the Backend.

        This is an optional property for the Backend if it targets being a
        compiler target. For classes of backends like circuit optimizers this
        doesn't apply and doesn't have to be defined.and will return None if a
        backend does not have a target.
        """
        if hasattr(self, '_target'):
            return self._target
        else:
            return None

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method that will return a :class:`~qiskit.providers.v2.Job` object
        that run circuits. Depending on the backend this may be either an async
        or sync call. It is the discretion of the provider to decide whether
        running should  block until the execution is finished or not. The Job
        class can handle either situation.
        """
        pass
