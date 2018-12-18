# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Provider for built-in simulator backends."""

from collections import OrderedDict
import logging

from qiskit.qiskiterror import QiskitError
from qiskit.providers import BaseProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import resolve_backend_name, filter_backends

from .qasm_simulator import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulatorPy
from .unitary_simulator import UnitarySimulatorPy


logger = logging.getLogger(__name__)

SIMULATORS = [
    QasmSimulatorPy,
    StatevectorSimulatorPy,
    UnitarySimulatorPy
]


class SimulatorsProvider(BaseProvider):
    """Provider for builtin simulators backends."""

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of builtin simulators backends.
        self._backends = self._verify_backends()

    def get_backend(self, name=None, **kwargs):
        backends = self._backends.values()

        # Special handling of the `name` parameter, to support alias resolution
        # and deprecated names.
        if name:
            try:
                resolved_name = resolve_backend_name(
                    name, backends,
                    self._deprecated_backend_names(),
                    {}
                )
                name = resolved_name
            except LookupError:
                raise QiskitBackendNotFoundError(
                    "The '{}' backend is not installed in your system.".format(name))

        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        backends = self._backends.values()

        # Special handling of the `name` parameter, to support alias resolution
        # and deprecated names.
        if name:
            try:
                resolved_name = resolve_backend_name(
                    name, backends,
                    self._deprecated_backend_names(),
                    {}
                )
                backends = [backend for backend in backends if
                            backend.name() == resolved_name]
            except LookupError:
                return []

        return filter_backends(backends, filters=filters, **kwargs)

    @staticmethod
    def _deprecated_backend_names():
        """Returns deprecated backend names."""
        return {
            'qasm_simulator_py': 'qasm_simulator',
            'statevector_simulator_py': 'statevector_simulator',
            'unitary_simulator_py': 'unitary_simulator',
            'local_qasm_simulator_py': 'qasm_simulator',
            'local_statevector_simulator_py': 'statevector_simulator',
            'local_unitary_simulator_py': 'unitary_simulator',
            'local_unitary_simulator': 'unitary_simulator',
            }

    def _verify_backends(self):
        """
        Return the builtin simulators backends in `BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:BaseBackend]: a dict of builtin simulators backend instances for
                the backends that could be instantiated, keyed by backend name.
        """
        ret = OrderedDict()
        for backend_cls in SIMULATORS:
            try:
                backend_instance = self._get_backend_instance(backend_cls)
                backend_name = backend_instance.name()
                ret[backend_name] = backend_instance
            except QiskitError as err:
                # Ignore backends that could not be initialized.
                logger.info('builtin simulators backend %s is not available: %s',
                            backend_cls, str(err))
        return ret

    def _get_backend_instance(self, backend_cls):
        """
        Return an instance of a backend from its class.

        Args:
            backend_cls (class): Backend class.
        Returns:
            BaseBackend: a backend instance.
        Raises:
            QiskitError: if the backend could not be instantiated.
        """
        # Verify that the backend can be instantiated.
        try:
            backend_instance = backend_cls(provider=self)
        except Exception as err:
            raise QiskitError('Backend %s could not be instantiated: %s' %
                              (backend_cls, err))

        return backend_instance

    def __str__(self):
        return 'BasicAer'
