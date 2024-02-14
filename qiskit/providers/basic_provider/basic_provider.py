# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Provider for basic simulator backends, formerly known as `BasicAer`."""

from __future__ import annotations

from collections.abc import Callable
from collections import OrderedDict
from typing import Type

import logging

from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends

from .basic_simulator import BasicSimulator


logger = logging.getLogger(__name__)

SIMULATORS = [BasicSimulator]


class BasicProvider(ProviderV1):
    """Provider for test simulators."""

    def __init__(self) -> None:
        super().__init__()

        # Populate the list of test backends (simulators)
        self._backends = self._verify_backends()

    def get_backend(self, name: str | None = None, **kwargs) -> Backend:
        return super().get_backend(name=name, **kwargs)

    def backends(
        self, name: str | None = None, filters: Callable | None = None, **kwargs
    ) -> list[Backend]:
        backends = self._backends.values()
        if name:
            available = [
                backend.name() if backend.version == 1 else backend.name for backend in backends
            ]
            if name not in available:
                raise QiskitBackendNotFoundError(
                    f"The '{name}' backend is not installed in your system."
                )
        return filter_backends(backends, filters=filters, **kwargs)

    def _verify_backends(self) -> OrderedDict[str, Backend]:
        """
        Return the test backends in `BACKENDS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:Backend]: a dict of test backend instances for
                the backends that could be instantiated, keyed by backend name.
        """
        ret = OrderedDict()
        for backend_cls in SIMULATORS:
            backend_instance = self._get_backend_instance(backend_cls)
            backend_name = backend_instance.name
            ret[backend_name] = backend_instance
        return ret

    def _get_backend_instance(self, backend_cls: Type[Backend]) -> Backend:
        """
        Return an instance of a backend from its class.

        Args:
            backend_cls (class): backend class.
        Returns:
            Backend: a backend instance.
        Raises:
            QiskitError: if the backend could not be instantiated.
        """
        # Verify that the backend can be instantiated.
        try:
            backend_instance = backend_cls(provider=self)
        except Exception as err:
            raise QiskitError(f"Backend {backend_cls} could not be instantiated: {err}") from err

        return backend_instance

    def __str__(self) -> str:
        return "BasicProvider"
