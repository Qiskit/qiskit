# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Provider for local backends."""

from collections import OrderedDict
import logging

from qiskit._qiskiterror import QISKitError
from qiskit.backends import BaseProvider
from qiskit.backends.providerutils import resolve_backend_name, filter_backends

from .qasm_simulator_cpp import CliffordSimulatorCpp, QasmSimulatorCpp
from .qasm_simulator_py import QasmSimulatorPy
from .statevector_simulator_cpp import StatevectorSimulatorCpp
from .statevector_simulator_py import StatevectorSimulatorPy
from .unitary_simulator_py import UnitarySimulatorPy


logger = logging.getLogger(__name__)

SDK_STANDARD_BACKENDS = [
    QasmSimulatorCpp,
    QasmSimulatorPy,
    StatevectorSimulatorCpp,
    StatevectorSimulatorPy,
    UnitarySimulatorPy,
    CliffordSimulatorCpp,
]


class LocalProvider(BaseProvider):
    """Provider for local backends."""

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of local backends.
        self._backends = self._verify_local_backends()

    def get_backend(self, name=None, **kwargs):
        backends = self._backends.values()

        # Special handling of the `name` parameter, to support alias resolution
        # and handling of groups.
        if name:
            try:
                resolved_names = resolve_backend_name(
                    name, backends,
                    self.grouped_backend_names(),
                    self.deprecated_backend_names(),
                    {}
                )
                name = resolved_names[0]
            except LookupError:
                pass

        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        backends = self._backends.values()

        # Special handling of the `name` parameter, to support alias resolution
        # and handling of groups.
        if name:
            try:
                resolved_names = resolve_backend_name(
                    name, backends,
                    self.grouped_backend_names(),
                    self.deprecated_backend_names(),
                    {}
                )
                backends = [backend for backend in backends if
                            backend.name() in resolved_names]
            except LookupError:
                return []

        return filter_backends(backends, filters=None, **kwargs)

    def grouped_backend_names(self):
        return {
            'local_qasm_simulator': ['local_qasm_simulator_cpp',
                                     'local_qasm_simulator_py',
                                     'local_clifford_simulator_cpp'],
            'local_statevector_simulator': ['local_statevector_simulator_cpp',
                                            'local_statevector_simulator_py'],
            'local_unitary_simulator': ['local_unitary_simulator_cpp',
                                        'local_unitary_simulator_py'],
            # TODO: restore after clifford simulator release
            # 'local_clifford_simulator': ['local_clifford_simulator_cpp']
            }

    def deprecated_backend_names(self):
        return {
            'local_qiskit_simulator': 'local_qasm_simulator_cpp',
            'wood_simulator': 'local_qasm_simulator_cpp',
            }

    def _verify_local_backends(self):
        """
        Return the local backends in `SDK_STANDARD_BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:BaseBackend]: a dict of the local backends instances for
                the backends that could be instantiated, keyed by backend name.
        """
        ret = OrderedDict()
        for backend_cls in SDK_STANDARD_BACKENDS:
            try:
                backend_instance = self._get_backend_instance(backend_cls)
                backend_name = backend_instance.configuration()['name']
                ret[backend_name] = backend_instance
            except QISKitError as err:
                # Ignore backends that could not be initialized.
                logger.info('local backend %s is not available: %s',
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
            QISKitError: if the backend could not be instantiated or does not
                provide a valid configuration containing a name.
        """
        # Verify that the backend can be instantiated.
        try:
            backend_instance = backend_cls(provider=self)
        except Exception as err:
            raise QISKitError('Backend %s could not be instantiated: %s' %
                              (backend_cls, err))

        # Verify that the instance has a minimal valid configuration.
        try:
            _ = backend_instance.configuration()['name']
        except (LookupError, TypeError):
            raise QISKitError('Backend %s has an invalid configuration')

        return backend_instance
