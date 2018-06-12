# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, bad-continuation

"""Provider for local backends."""
import logging

from qiskit._qiskiterror import QISKitError
from qiskit.backends import BaseProvider
from .qasm_simulator_cpp import CliffordSimulatorCpp, QasmSimulatorCpp
from .qasm_simulator_py import QasmSimulatorPy
from .qasm_simulator_projectq import QasmSimulatorProjectQ
from .statevector_simulator_cpp import StatevectorSimulatorCpp
from .statevector_simulator_py import StatevectorSimulatorPy
from .unitary_simulator_py import UnitarySimulatorPy


logger = logging.getLogger(__name__)

SDK_STANDARD_BACKENDS = [
    QasmSimulatorCpp,
    QasmSimulatorPy,
    QasmSimulatorProjectQ,
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
        self.backends = self._verify_local_backends()

    def get_backend(self, name):
        return self.backends[name]

    def available_backends(self, filters=None):
        # pylint: disable=arguments-differ
        backends = self.backends

        filters = filters or {}
        for key, value in filters.items():
            backends = {name: instance for name, instance in backends.items()
                        if instance.configuration.get(key) == value}
        return list(backends.values())

    def aliased_backend_names(self):
        return {
            'local_qasm_simulator': ['local_qasm_simulator_cpp',
                                     'local_qasm_simulator_projectq',
                                     'local_qasm_simulator_py',
                                     'local_clifford_simulator_cpp'],
            'local_statevector_simulator': ['local_statevector_simulator_cpp',
                                            'local_statevector_simulator_projectq',
                                            'local_statevector_simulator_py'],
            'local_unitary_simulator': ['local_unitary_simulator_cpp',
                                        'local_unitary_simulator_py']
            # TODO: restore after clifford simulator release
            # 'local_clifford_simulator': ['local_clifford_simulator_cpp']
            }

    def deprecated_backend_names(self):
        return {
            'local_qiskit_simulator': 'local_qasm_simulator_cpp',
            'wood_simulator': 'local_qasm_simulator_cpp',
            }

    @classmethod
    def _verify_local_backends(cls):
        """
        Return the local backends in `SDK_STANDARD_BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:BaseBackend]: a dict of the local backends instances for
                the backends that could be instantiated, keyed by backend name.
        """
        ret = {}
        for backend_cls in SDK_STANDARD_BACKENDS:
            try:
                backend_instance = cls._get_backend_instance(backend_cls)
                backend_name = backend_instance.configuration['name']
                ret[backend_name] = backend_instance
            except QISKitError as e:
                # Ignore backends that could not be initialized.
                logger.info('local backend %s is not available: %s',
                            backend_cls, str(e))
        return ret

    @classmethod
    def _get_backend_instance(cls, backend_cls):
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
            backend_instance = backend_cls()
        except Exception as err:
            raise QISKitError('Backend %s could not be instantiated: %s' %
                              (cls, err))

        # Verify that the instance has a minimal valid configuration.
        try:
            _ = backend_instance.configuration['name']
        except (LookupError, TypeError):
            raise QISKitError('Backend %s has an invalid configuration')

        return backend_instance
