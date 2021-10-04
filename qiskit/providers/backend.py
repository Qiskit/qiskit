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

# pylint: disable=invalid-name

"""Backend abstract interface for providers."""


from abc import ABC
from abc import abstractmethod
from typing import List, Union

import numpy as np

from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Gate


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
    The first are the attributes of the class itself. These should be used to
    defined the immutable characteristics of the backend. The ``options``
    attribute of the backend is used to contain the dynamic user configurable
    options of the backend. It should be used more for runtime options
    that configure how the backend is used. For example, something like a
    ``shots`` field for a backend that runs experiments which would contain an
    int for how many shots to execute. The ``properties`` attribute is
    optionally defined :class:`~qiskit.providers.models.BackendProperties`
    object and is used to return measured properties, or properties
    of a backend that may change over time. The simplest example of this would
    be a version string, which will change as a backend is updated, but also
    could be something like noise parameters for backends that run experiments.

    This first version of the Backend abstract class is written to be mostly
    backwards compatible with the legacy providers interface. This includes reusing
    the model objects :class:`~qiskit.providers.models.BackendProperties` and
    :class:`~qiskit.providers.models.BackendConfiguration`. This was done to
    ease the transition for users and provider maintainers to the new versioned providers.
    Expect, future versions of this abstract class to change the data model and
    interface.
    """

    version = 1

    def __init__(self, configuration, provider=None, **fields):
        """Initialize a backend class

        Args:
            configuration (BackendConfiguration): A backend configuration
                object for the backend object.
            provider (qiskit.providers.Provider): Optionally, the provider
                object that this Backend comes from.
            fields: kwargs for the values to use to override the default
                options.
        Raises:
            AttributeError: if input field not a valid options

        ..
            This next bit is necessary just because autosummary generally won't summarise private
            methods; changing that behaviour would have annoying knock-on effects through all the
            rest of the documentation, so instead we just hard-code the automethod directive.

        In addition to the public abstract methods, subclasses should also implement the following
        private methods:

        .. automethod:: _default_options
        """
        self._configuration = configuration
        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError("Options field %s is not valid for this backend" % field)
            self._options.update_config(**fields)

    @classmethod
    @abstractmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
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
            if not hasattr(self._options, field):
                raise AttributeError("Options field %s is not valid for this backend" % field)
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
        return BackendStatus(
            backend_name=self.name(),
            backend_version="1",
            operational=True,
            pending_jobs=0,
            status_msg="",
        )

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
        return f"<{self.__class__.__name__}('{self.name()}')>"

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

        This method that will return a :class:`~qiskit.providers.Job` object
        that run circuits. Depending on the backend this may be either an async
        or sync call. It is the discretion of the provider to decide whether
        running should  block until the execution is finished or not. The Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or list): An individual or a
                list of :class:`~qiskit.circuits.QuantumCircuit` or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
                For legacy providers migrating to the new versioned providers,
                provider interface a :class:`~qiskit.qobj.QasmQobj` or
                :class:`~qiskit.qobj.PulseQobj` objects should probably be
                supported too (but deprecated) for backwards compatibility. Be
                sure to update the docstrings of subclasses implementing this
                method to document that. New provider implementations should not
                do this though as :mod:`qiskit.qobj` will be deprecated and
                removed along with the legacy providers interface.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """
        pass


class BackendV2(Backend, ABC):
    """Abstract class for Backends

    This abstract class is to be used for all Backend objects created by a
    provider. This version differs from earlier abstract Backend classes in
    that the configuration attribute no longer exists instead attributes
    exposing equivalent required immutable properties of the backend device
    are added. For example ``backend.configuration().n_qubits`` is accessible
    from ``backend.num_qubits`` now.

    The ``options`` attribute of the backend is used to contain the dynamic
    user configurable options of the backend. It should be used more for
    runtime options that configure how the backend is used. For example,
    something like a ``shots`` field for a backend that runs experiments which
    would contain an int for how many shots to execute.

    If migrating a provider from :class:`~qiskit.providers.BackendV1` or
    :class:`~qiskit.providers.BaseBackend` one thing to keep in mind is for
    backwards compatibility you might need to add a configuration method that
    will build a :class:`~qiskit.providers.models.BackendConfiguration` object
    and :class:`~qiskit.providers.models.BackendProperties` from the attributes
    defined in this class for backwards compatibility.
    """

    version = 2

    def __init__(self, provider, **fields):
        self._options = self._default_options()
        self._provider = provider
        self._coupling_map = None
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError("Options field %s is not valid for this backend" % field)
            self._options.update_config(**fields)

    @property
    def instructions(self) -> List[Gate]:
        """A list of :class:`~qiskit.circuit.Instruction` instances that the backend supports."""
        return list(self.target.instructions)

    @property
    def instruction_names(self) -> List[str]:
        """A list of instruction names that the backend supports."""
        return list(self.target.instruction_names)

    @property
    @abstractmethod
    def target(self):
        """A :class:`qiskit.transpiler.Target` object for the backend."""
        pass

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits the backend has."""
        return self.target.num_qubits

    @property
    def coupling_map(self):
        """Return the :class:`~qiskit.transpiler.CouplingMap` object"""
        return self.target.coupling_map()

    @property
    def instruction_durations(self):
        """Return the :class:`~qiskit.transpiler.InstructionDurations` object."""
        return self.target.durations()

    @property
    @abstractmethod
    def conditional(self) -> bool:
        """Return bool whether the target can execute gates with classical
        conditions."""
        pass

    @property
    @abstractmethod
    def max_shots(self) -> int:
        """Return the maximum number of shots supported by the backend.

        If there is no limit this will return None
        """
        pass

    @property
    @abstractmethod
    def max_circuits(self):
        """The maximum number of circuits (or Pulse schedules) that can be
        run in a single job.

        If there is no limit this will return None
        """
        pass

    @classmethod
    @abstractmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """
        pass

    def t1(self, qubit: Union[int, List[int]]) -> Union[float, np.array]:
        """Return the T1 time of a given qubit

        Args:
            qubit: The qubit index to get the T1 time for. If
                a list is specified the output will be a list with the
                T1 time for the specified qubits in the same order.

        Returns:
            t1: the T1 time for the specified qubit(s) in seconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                t1 time for a qubit
        """
        raise NotImplementedError

    def t2(self, qubit: Union[int, List[int]]) -> Union[float, np.array]:
        """Return the T2 time of a given qubit

        Args:
            qubit: The qubit index or indices to get the T2 time for. If
                a list is specified the output will be a list with the
                T2 time for the specified qubits in the same order.

        Returns:
            t2: the T2 time for the specified qubit(s) in seconds

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                t2 time for a qubit
        """
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """Return the qubit drive channel timestep in seconds

        Returns:
            dt: The qubit drive channel timestep in seconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                qubit drive channel timestep
        """
        raise NotImplementedError

    @property
    def dtm(self) -> float:
        """Return the measurement drive channel timestep in nanoseconds

        Returns:
            dtm: The measurement drive channel timestep in nanoseconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement drive channel timestep
        """
        raise NotImplementedError

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
            if not hasattr(self._options, field):
                raise AttributeError("Options field %s is not valid for this " "backend" % field)
        self._options.update_options(**fields)

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method that will return a :class:`~qiskit.providers.Job` object
        that run circuits. Depending on the backend this may be either an async
        or sync call. It is the discretion of the provider to decide whether
        running should  block until the execution is finished or not. The Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or list): An individual or a
                list of :class:`~qiskit.circuits.QuantumCircuit` or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
                For legacy providers migrating to the new versioned providers,
                provider interface a :class:`~qiskit.qobj.QasmQobj` or
                :class:`~qiskit.qobj.PulseQobj` objects should probably be
                supported too (but deprecated) for backwards compatibility. Be
                sure to update the docstrings of subclasses implementing this
                method to document that. New provider implementations should not
                do this though as :mod:`qiskit.qobj` will be deprecated and
                removed along with the legacy providers interface.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """
        pass
