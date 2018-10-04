# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for a single IBMQ account."""

from collections import OrderedDict

from IBMQuantumExperience import IBMQuantumExperience

from qiskit._util import _camel_case_to_snake_case
from qiskit.backends import BaseProvider
from qiskit.backends.ibmq.ibmqbackend import IBMQBackend
from qiskit.backends.providerutils import filter_backends


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

        # Get a connection to IBMQuantumExperience.
        self.credentials = credentials
        self._api = self._authenticate(self.credentials)
        self._ibm_provider = ibmq_provider

        # Populate the list of remote backends.
        self._backends = self._discover_remote_backends()

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        backends = self._backends.values()

        if name:
            kwargs['name'] = name

        return filter_backends(backends, filters=filters, **kwargs)

    @classmethod
    def _authenticate(cls, credentials):
        """Authenticate against the IBMQuantumExperience API.

        Args:
            credentials (Credentials): Quantum Experience or IBMQ credentials.

        Returns:
            IBMQuantumExperience.IBMQuantumExperience.IBMQuantumExperience:
                instance of the IBMQuantumExperience API.
        Raises:
            ConnectionError: if the authentication resulted in error.
        """
        try:
            config_dict = {
                'url': credentials.url,
            }
            if credentials.proxies:
                config_dict['proxies'] = credentials.proxies
            return IBMQuantumExperience(credentials.token, config_dict,
                                        credentials.verify)
        except Exception as ex:
            root_exception = ex
            if 'License required' in str(ex):
                # For the 401 License required exception from the API, be
                # less verbose with the exceptions.
                root_exception = None
            raise ConnectionError("Couldn't connect to IBMQuantumExperience server: {0}"
                                  .format(ex)) from root_exception

    @classmethod
    def _parse_backend_configuration(cls, config):
        """Parse a backend configuration returned by IBMQuantumConfiguration.

        Args:
            config (dict): raw configuration as returned by
                IBMQuantumExperience.

        Returns:
            dict: parsed configuration.
        """
        edited_config = {
            'local': False
        }

        for key in config.keys():
            new_key = _camel_case_to_snake_case(key)
            if new_key not in ['id', 'serial_number', 'topology_id',
                               'status']:
                edited_config[new_key] = config[key]

        return edited_config

    def _discover_remote_backends(self):
        """Return the remote backends available.

        Returns:
            dict[str:IBMQBackend]: a dict of the remote backend instances,
                keyed by backend name.
        """
        ret = OrderedDict()
        configs_list = self._api.available_backends()
        for raw_config in configs_list:
            config = self._parse_backend_configuration(raw_config)
            ret[config['name']] = IBMQBackend(configuration=config,
                                              provider=self._ibm_provider,
                                              credentials=self.credentials,
                                              api=self._api)

        return ret

    def __eq__(self, other):
        return self.credentials == other.credentials
