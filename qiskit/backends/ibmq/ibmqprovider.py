# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for remote IbmQ backends."""
from IBMQuantumExperience import IBMQuantumExperience

from qiskit._util import _camel_case_to_snake_case
from qiskit.backends.baseprovider import BaseProvider
from qiskit.backends.ibmq.ibmqbackend import IBMQBackend
from qiskit._util import _parse_ibmq_credentials


class IBMQProvider(BaseProvider):
    """Provider for remote IbmQ backends."""
    def __init__(self, token, url='https://quantumexperience.ng.bluemix.net/api',
                 hub=None, group=None, project=None, proxies=None, verify=True):
        super().__init__()

        url = _parse_ibmq_credentials(url, hub, group, project)

        # Get a connection to IBMQuantumExperience.
        self._api = self._authenticate(token, url, proxies=proxies, verify=verify)

        # Populate the list of remote backends.
        self.backends = self._discover_remote_backends()

        # authentication attributes, which uniquely identify the provider instance
        self._token = token
        self._url = url
        self._proxies = proxies
        self._verify = verify

    def get_backend(self, name):
        return self.backends[name]

    def available_backends(self):
        """Get a list of available backends from the IBMQ provider.

        Returns:
            list[IBMQBackend]: a list of backend instances available
            from the IBMQ provider.
        """
        # pylint: disable=arguments-differ
        return list(self.backends.values())

    def grouped_backend_names(self):
        return {}

    def deprecated_backend_names(self):
        return {
            'ibmqx_qasm_simulator': 'ibmq_qasm_simulator',
            'ibmqx_hpc_qasm_simulator': 'ibmq_qasm_simulator',
            'real': 'ibmqx1'
            }

    def aliased_backend_names(self):
        return {
            'ibmq_5_yorktown': 'ibmqx2',
            'ibmq_5_tenerife': 'ibmqx4',
            'ibmq_16_rueschlikon': 'ibmqx5',
            'ibmq_20_austin': 'QS1_1'
            }

    @classmethod
    def _authenticate(cls, token, url, proxies=None, verify=True):
        """
        Authenticate against the IBMQuantumExperience API.

        Returns:
            IBMQuantumExperience.IBMQuantumExperience.IBMQuantumExperience:
                instance of the IBMQuantumExperience API.
        Raises:
            ConnectionError: if the authentication resulted in error.
        """
        try:
            config_dict = {
                'url': url,
            }
            if proxies:
                config_dict['proxies'] = proxies
            return IBMQuantumExperience(token, config_dict, verify)
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
        """
        Parse a backend configuration returned by IBMQuantumConfiguration.

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
        """
        Return the remote backends available.

        Returns:
            dict[str:IBMQBackend]: a dict of the remote backend instances,
                keyed by backend name.
        """
        ret = {}
        configs_list = self._api.available_backends()
        for raw_config in configs_list:
            config = self._parse_backend_configuration(raw_config)
            ret[config['name']] = IBMQBackend(configuration=config, api=self._api)

        return ret

    def __eq__(self, other):
        try:
            equality = (self._token == other._token and self._url == other._url)
        except AttributeError:
            equality = False
        return equality
