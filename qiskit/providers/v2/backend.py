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

"""Backend abstract interface for providers."""


from abc import ABC
from abc import abstractmethod

from qiskit.providers.models.backendstatus import BackendStatus


class Backend:
    """Base common type for all versioned Backend abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class BackendV1(Backend, ABC):
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

    This first version of the Backend abstract class is written to be mostly
    backwards compatible with the v1 providers interface. This was done to ease
    the transition for users and provider maintainers from v1 to v2. Expect,
    future versions of this abstract class to change the data model and
    interface.
    """
    version = 1

    def __init__(self, configuration, provider=None, **fields):
        """Initialize a backend class

        Args:
            configuration (BackendConfiguration): A backend configuration
                object for the backend object.
            provider (qiskit.providers.v2.Provider): Optionally, the provider
                object that this Backend comes from.
            fields: kwargs for the values to use to override the default
                options.
        Raises:
            AttributeError: if input field not a valid options
        """
        self._configuration = configuration
        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError(
                        "Options field %s is not valid for this backend" % field)
            self._options.update_config(**fields)

    @classmethod
    @abstractmethod
    def _default_options(cls):
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
            if not hasattr(field, self._option):
                raise AttributeError(
                    "Options field %s is not valid for this "
                    "backend" % field)
        self._options.update_options(**fields)

    def configuration(self):
        """Return the backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        return self._configuration

    def properties(self):
        """Return the backend properties.

        Returns:
            BackendProperties: the configuration for the backend. If the backend
            does not support properties, it returns ``None``.
        """
        return None

    def provider(self):
        """Return the backend Provider.

        Returns:
            Provider: the Provider responsible for the backend.
        """
        return self._provider

    def status(self):
        """Return the backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(backend_name=self.name(),
                             backend_version='1',
                             operational=True,
                             pending_jobs=0,
                             status_msg='')

    def name(self):
        """Return the backend name.

        Returns:
            str: the name of the backend.
        """
        return self._configuration.backend_name

    def __str__(self):
        return self.name()

    def __repr__(self):
        """Official string representation of a Backend.

        Note that, by Qiskit convention, it is consciously *not* a fully valid
        Python expression. Subclasses should provide 'a string of the form
        <...some useful description...>'. [0]

        [0] https://docs.python.org/3/reference/datamodel.html#object.__repr__
        """
        return "<{}('{}')>".format(self.__class__.__name__, self.name())

    @property
    def options(self):
        """Return the options for the backend

        The options of a backend are the dynamic parameters defining
        how the backend is used. These are used to control the :meth:`run`
        method.
        """
        return self._options

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method that will return a :class:`~qiskit.providers.v2.Job` object
        that run circuits. Depending on the backend this may be either an async
        or sync call. It is the discretion of the provider to decide whether
        running should  block until the execution is finished or not. The Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or list): An individual or a
                list of :class:`~qiskit.circuits.QuantumCircuit` or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
                For v1 providers migrating to this first version of a v2
                provider interface a :class:`~qiskit.qobj.QasmQobj` or
                :class:`~qiskit.qobj.PulseQobj` object. These will be
                deprecated and removed from terra in the future.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """
        pass
