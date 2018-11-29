# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Provider for aer backends."""

from collections import OrderedDict
import logging

from qiskit._qiskiterror import QISKitError
from qiskit.backends import BaseProvider
from qiskit.backends.exceptions import QiskitBackendNotFoundError
from qiskit.backends.providerutils import resolve_backend_name, filter_backends

from .qasm_simulator import CliffordSimulator, QasmSimulator
from .qasm_simulator_py import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulator
from .statevector_simulator_py import StatevectorSimulatorPy
from .unitary_simulator_py import UnitarySimulatorPy


logger = logging.getLogger(__name__)

AER_STANDARD_BACKENDS = [
    QasmSimulator,
    QasmSimulatorPy,
    StatevectorSimulator,
    StatevectorSimulatorPy,
    UnitarySimulatorPy,
    CliffordSimulator,
]


class AerProvider(BaseProvider):
    """Provider for aer backends."""

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of aer backends.
        self._backends = self._verify_aer_backends()

    def get_backend(self, name=None, **kwargs):
        backends = self._backends.values()

        # Special handling of the `name` parameter, to support alias resolution
        # and deprecated names.
        if name:
            try:
                resolved_name = resolve_backend_name(
                    name, backends,
                    self.deprecated_backend_names(),
                    {},
                    self._alternative_py_backend_names()
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
                    self.deprecated_backend_names(),
                    {},
                    self._alternative_py_backend_names()
                )
                backends = [backend for backend in backends if
                            backend.name() == resolved_name]
            except LookupError:
                return []

        return filter_backends(backends, filters=filters, **kwargs)

    @staticmethod
    def deprecated_backend_names():
        """Returns deprecated backend names."""
        return {
            'local_qasm_simulator_cpp': 'qasm_simulator',
            'local_qasm_simulator_py': 'qasm_simulator_py',
            'local_statevector_simulator_cpp': 'statevector_simulator',
            'local_statevector_simulator_py': 'statevector_simulator_py',
            'local_unitary_simulator_py': 'unitary_simulator_py',
            'local_qiskit_simulator': 'qasm_simulator',
            'local_qasm_simulator': 'qasm_simulator',
            'local_statevector_simulator': 'statevector_simulator',
            'local_unitary_simulator': 'unitary_simulator_py',
            'unitary_simulator': 'unitary_simulator_py'
            }

    def _verify_aer_backends(self):
        """
        Return the aer backends in `AER_STANDARD_BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:BaseBackend]: a dict of aer backend instances for
                the backends that could be instantiated, keyed by backend name.
        """
        ret = OrderedDict()
        for backend_cls in AER_STANDARD_BACKENDS:
            try:
                backend_instance = self._get_backend_instance(backend_cls)
                backend_name = backend_instance.name()
                ret[backend_name] = backend_instance
            except QISKitError as err:
                # Ignore backends that could not be initialized.
                logger.info('aer backend %s is not available: %s',
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
            QISKitError: if the backend could not be instantiated.
        """
        # Verify that the backend can be instantiated.
        try:
            backend_instance = backend_cls(provider=self)
        except Exception as err:
            raise QISKitError('Backend %s could not be instantiated: %s' %
                              (backend_cls, err))

        return backend_instance

    def __str__(self):
        return 'Aer'

    @staticmethod
    def _alternative_py_backend_names():
        """Return Python alternatives to C++ simulators."""
        return {
            'qasm_simulator': 'qasm_simulator_py',
            'statevector_simulator': 'statevector_simulator_py'
        }
