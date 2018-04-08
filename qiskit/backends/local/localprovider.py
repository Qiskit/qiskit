# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Provider for local backends."""
import logging

from qiskit import QISKitError
from qiskit.backends._projectq_simulator import ProjectQSimulator
from qiskit.backends._qasmsimulator import QasmSimulator
from qiskit.backends._qiskit_cpp_simulator import (QISKitCppSimulator,
                                                   CliffordCppSimulator)
from qiskit.backends._sympy_qasmsimulator import SympyQasmSimulator
from qiskit.backends._unitarysimulator import UnitarySimulator
from qiskit.backends.baseprovider import BaseProvider


logger = logging.getLogger(__name__)

SDK_STANDARD_BACKENDS = [
    CliffordCppSimulator,
    ProjectQSimulator,
    QasmSimulator,
    QISKitCppSimulator,
    SympyQasmSimulator,
    UnitarySimulator
]


class LocalProvider(BaseProvider):
    """Provider for local backends."""
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of local backends.
        self.backends = self._verify_local_backends()

    def get_backend(self, name):
        return self.backends[name]()

    def available_backends(self):
        return list(self.backends.keys())

    @classmethod
    def _verify_local_backends(cls):
        """
        Return the local backends in `SDK_STANDARD_BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict: (str: class) a dict of the local backends classes that can
                be instantiated, keyed by backend name.
        """
        ret = {}
        for backend_cls in SDK_STANDARD_BACKENDS:
            try:
                backend_name = cls._get_backend_name(backend_cls)
                ret[backend_name] = backend_cls
            except QISKitError as e:
                # Ignore backends that could not be initialized.
                logger.info('local backend %s is not available: %s',
                            backend_cls, str(e))
        return ret

    @classmethod
    def _get_backend_name(cls, backend_cls):
        """
        Return the name of the backend, by instantiating it and reading the
        name from its configuration.

        Args:
            backend_cls (class): Backend class.
        Returns:
            str: the name of the backend.
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
            backend_name = backend_instance.configuration['name']
        except (LookupError, TypeError):
            raise QISKitError('Backend %s has an invalid configuration')

        return backend_name
