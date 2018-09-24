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

    def backend(self, name=None, **kwargs):
        try:
            return self.backends(name, **kwargs)[0]
        except IndexError:
            raise KeyError(name)

    def backends(self, name=None, filters=None, **kwargs):
        """Return the available backends matching the specified filtering.

        Return the list of available backends from this provider, optionally
        filtering the results by the backends' `configuration` or `status`
        attributes, or from a boolean callable. The criteria for filtering can
        be specified via `**kwargs` or as a callable via `filters`, and the
        backends must fulfill all specified conditions.

        For example::

            backends(simulator=True,
                     operational=True,
                     filters=lambda x: 'snapshot' in x.configuration()['basis_gates'])

        Will return all the operational simulators that support 'snapshot'.

        Args:
            name (str): name of the backend.
            filters (callable): filtering conditions as a callable
                (`BaseBackend` -> bool). For example::

                    backends(filters=lambda x: x.configuration()['n_qubits'] > 5)

            **kwargs (dict): dict of 'criteria': value pairs, that will be
                matched against the backend's `configuration` or `status`
                attributes. For example::

                    backends(local=False, operational=True)

        Returns:
            list[BaseBackend]: a list of backend instances matching the
                conditions.
        """
        def _match_all(dict_, criteria):
            """Return True if all items in criteria matches items in dict_."""
            return all(dict_.get(key_) == value_ for
                       key_, value_ in criteria.items())

        backends = self._backends_list()

        # Special handling of the `name` parameter, to support alias resolution
        # and handling of groups.
        if name:
            try:
                resolved_names = self.resolve_backend_name(name)
                backends = [backend for backend in backends if
                            backend.name() in resolved_names]
            except LookupError:
                return []

        # Inspect the backends to decide which filters belong to
        # backend.configuration and which ones to backend.status, as it does
        # not involve querying the API.
        configuration_filters = {}
        status_filters = {}
        for key, value in kwargs.items():
            if all(key in backend.configuration() for backend in backends):
                configuration_filters[key] = kwargs[key]
            else:
                status_filters[key] = kwargs[key]

        # 1. Apply backend.configuration filtering.
        if configuration_filters:
            backends = [b for b in backends if
                        _match_all(b.configuration(), configuration_filters)]

        # 2. Apply backend.status filtering (it involves one API call for
        # each backend).
        if status_filters:
            backends = [b for b in backends if
                        _match_all(b.status(), status_filters)]

        # 3. Apply acceptor filter.
        backends = list(filter(filters, backends))

        return backends
