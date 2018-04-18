# -*- coding: utf-8 -*-

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

"""Provider for remote IbmQ backends."""
from IBMQuantumExperience import IBMQuantumExperience

from qiskit._util import _snake_case_to_camel_case
from qiskit.backends.baseprovider import BaseProvider
from qiskit.backends.ibmq.ibmqbackend import IBMQBackend

_ALIASED_BACKENDS = {
    'local_qasm_simulator': ['local_qasm_simulator_cpp',
                             'local_qasm_simulator_projectq',
                             'local_qasm_simulator_py'],
    'local_statevector_simulator': ['local_statevector_simulator_cpp',
                                    'local_statevector_simulator_projectq',
                                    'local_statevector_simulator_py',
                                    'local_statevector_simulator_sympy'],
    'local_unitary_simulator': ['local_unitary_simulator_cpp',
                                'local_unitary_simulator_py',
                                'local_unitary_simulator_sympy'],
    # FIXME: uncomment after API fix: online simulator names should change
    # 'ibmq_qasm_simulator': ['ibmq_qasm_simulator',
    #                        'ibmq_qasm_simulator_hpc'],
    'local_clifford_simulator': ['local_clifford_simulator_cpp']
}
"""
dict (alias_name: backend_names(list))

Dict that defines alias names, usually shorter names for referring to
the backends.

If an alias key is used, the corresponding backend will be chosen in order
of priority from the value list, depending on availability.
"""

_DEPRECATED_BACKENDS = {
    'local_qiskit_simulator': 'local_qasm_simulator_cpp',
    'wood_simulator': 'local_qasm_simulator_cpp',
    # FIXME: uncomment after API fix: online simulator names should change
    # 'ibmqx_qasm_simulator': 'ibmq_qasm_simulator',
    # 'ibmqx_hpc_qasm_simulator': 'ibmq_qasm_simulator_hpc',
    'real': 'ibmqx1'
}
"""
dict (deprecated_name: backend_name)

Dict that stores the current name for all deprecated backends.
These will be removed in future releases.
"""


class IBMQProvider(BaseProvider):
    """Provider for remote IbmQ backends."""
    def __init__(self, token, url,
                 hub=None, group=None, project=None, proxies=None, verify=True):
        super().__init__()

        # Get a connection to IBMQuantumExperience.
        self._api = self._authenticate(token, url,
                                       hub, group, project, proxies, verify)

        # Populate the list of remote backends.
        self.backends = self._discover_remote_backends()

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
            # FIXME: uncomment after API fix: online simulator names should change
            # 'ibmq_qasm_simulator': ['ibmq_qasm_simulator',
            #                         'ibmq_qasm_simulator_hpc']
            }

    def deprecated_backend_names(self):
        return {
            # FIXME: uncomment after API fix: online simulator names should change
            # 'ibmq_qasm_simulator': 'ibmq_qasm_simulator',
            # 'ibmq_qasm_simulator_hpc': 'ibmq_qasm_simulator_hpc',
            'real': 'ibmqx1'
            }

    @classmethod
    def _authenticate(cls, token, url,
                      hub=None, group=None, project=None, proxies=None,
                      verify=True):
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
            # Only append hub/group/project if they are different than None.
            if all([hub, group, project]):
                config_dict.update({
                    'hub': hub,
                    'group': group,
                    'project': project
                })
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
            new_key = _snake_case_to_camel_case(key)
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

    def _resolve_backend_name(self, backend_name):
        """Resolve backend name from a possible short alias or a deprecated name.

        The alias will be chosen in order of priority, depending on availability.

        Args:
            backend (str): name of backend to resolve

        Returns:
            str: name of resolved backend

        Raises:
            LookupError: if backend cannot be resolved through registered names,
            nor aliases, nor deprecated names
        """
        resolved_backend = ""
        if backend in _REGISTERED_BACKENDS:
            resolved_backend = backend
        elif backend in _ALIASED_BACKENDS:
            available_aliases = [b for b in _ALIASED_BACKENDS[backend]
                                 if b in _REGISTERED_BACKENDS]
            if available_aliases:
                resolved_backend = available_aliases[0]
        elif backend in _DEPRECATED_BACKENDS:
            resolved_backend = _DEPRECATED_BACKENDS[backend]
            logger.warning('WARNING: %s is deprecated. Use %s.', backend, resolved_backend)
        # FIXME: remove after API fix: online simulator names should change
        if backend == 'ibmq_qasm_simulator':
            resolved_backend = 'ibmqx_qasm_simulator'
        if backend == 'ibmq_qasm_simulator_hpc':
            resolved_backend = 'ibmqx_hpc_qasm_simulator'

        if resolved_backend not in _REGISTERED_BACKENDS:
            raise LookupError('backend "{}" is not available'.format(backend))

    return resolved_backend
