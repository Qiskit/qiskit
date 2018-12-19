# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for a single IBMQ account."""

import logging
from collections import OrderedDict

from marshmallow import ValidationError

from qiskit.providers import BaseProvider
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.providerutils import filter_backends

from .api import IBMQConnector
from .ibmqbackend import IBMQBackend


logger = logging.getLogger(__name__)


class IBMQSingleProvider(BaseProvider):
    """Provider for single IBMQ accounts.

    Note: this class is not part of the public API and is not guaranteed to be
    present in future releases.
    """
    def __init__(self, credentials, ibmq_provider):
        """
        Args:
            credentials (Credentials): Quantum Experience or IBMQ credentials.
            ibmq_provider (IBMQProvider): IBMQ main provider.
        """
        super().__init__()

        # Get a connection to IBMQ.
        self.credentials = credentials
        self._api = self._authenticate(self.credentials)
        self._ibm_provider = ibmq_provider

        # Populate the list of remote backends.
        self._backends = self._discover_remote_backends()

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        backends = self._backends.values()

        if name:
            kwargs['backend_name'] = name

        return filter_backends(backends, filters=filters, **kwargs)

    @classmethod
    def _authenticate(cls, credentials):
        """Authenticate against the IBMQ API.

        Args:
            credentials (Credentials): Quantum Experience or IBMQ credentials.

        Returns:
            IBMQConnector: instance of the IBMQConnector.
        Raises:
            ConnectionError: if the authentication resulted in error.
        """
        try:
            config_dict = {
                'url': credentials.url,
            }
            if credentials.proxies:
                config_dict['proxies'] = credentials.proxies
            return IBMQConnector(credentials.token, config_dict,
                                 credentials.verify)
        except Exception as ex:
            root_exception = ex
            if 'License required' in str(ex):
                # For the 401 License required exception from the API, be
                # less verbose with the exceptions.
                root_exception = None
            raise ConnectionError("Couldn't connect to IBMQ server: {0}"
                                  .format(ex)) from root_exception

    def _discover_remote_backends(self):
        """Return the remote backends available.

        Returns:
            dict[str:IBMQBackend]: a dict of the remote backend instances,
                keyed by backend name.
        """
        ret = OrderedDict()
        configs_list = self._api.available_backends()
        for raw_config in configs_list:
            try:
                config = BackendConfiguration.from_dict(raw_config)
                ret[config.backend_name] = IBMQBackend(
                    configuration=config,
                    provider=self._ibm_provider,
                    credentials=self.credentials,
                    api=self._api)
            except ValidationError as ex:
                logger.warning(
                    'Remote backend "%s" could not be instantiated due to an '
                    'invalid config: %s',
                    raw_config.get('backend_name',
                                   raw_config.get('name', 'unknown')),
                    ex)

        return ret

    def __eq__(self, other):
        return self.credentials == other.credentials
