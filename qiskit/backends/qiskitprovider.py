# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider implementing functionality specific for Aer and IBMQ."""

from collections import OrderedDict

from qiskit import QISKitError
from qiskit.backends.baseprovider import BaseProvider


class QiskitProvider(BaseProvider):
    """
    Provider with features not part of the generic BaseProvider, but used
    frequently in Terra.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._backends = OrderedDict()

    def _backends_list(self):
        return list(self._backends.values())

    def backends(self, filters=None, **kwargs):
        """Return the available backends.

        Note:
            If two or more providers share similar backend names, only the backends
            belonging to the first registered provider will be returned.

        Args:
            filters (dict or callable): filtering conditions.
                each will either pass through, or be filtered out:

                1) dict: {'criteria': value}
                    the criteria can be over backend's `configuration` or `status`
                    e.g. {'local': False, 'simulator': False, 'operational': True}

                2) callable: BaseBackend -> bool
                    e.g. lambda x: x.configuration()['n_qubits'] > 5

        Returns:
            list[BaseBackend]: a list of backend instances available
                from all the providers.

        Raises:
            QISKitError: if passing filters that is neither dict nor callable
        """
        # TODO: alias resolution for 'name'
        # TODO: 'filter' ergonomics
        # TODO: filter always queries for 'status', slow for IBMQ

        backends = self._backends_list()

        if filters is not None:
            if isinstance(filters, dict):
                # exact match filter:
                # e.g. {'n_qubits': 5, 'operational': True}
                for key, value in filters.items():
                    backends = [instance for instance in backends
                                if instance.configuration().get(key) == value
                                or instance.status().get(key) == value]
            elif callable(filters):
                # acceptor filter: accept or reject a specific backend
                # e.g. lambda x: x.configuration()['n_qubits'] > 5
                accepted_backends = []
                for backend in backends:
                    try:
                        if filters(backend) is True:
                            accepted_backends.append(backend)
                    except Exception:  # pylint: disable=broad-except
                        pass
                backends = accepted_backends
            else:
                raise QISKitError('backend filters must be either dict or callable.')

        return backends
