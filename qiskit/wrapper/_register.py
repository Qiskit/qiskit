# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Internal register functions"""

import os
import warnings
import importlib.util
from qiskit._qiskiterror import QISKitError
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from qiskit._util import (_provider_name_from_url, _parse_ibmq_credentials)

# Default provider used by the rest of the functions on this module. Please
# note that this is a global object.
_DEFAULT_PROVIDER = DefaultQISKitProvider()

REGISTER_CALLED = 0


def _register(token=None, url='https://quantumexperience.ng.bluemix.net/api',
              proxies=None, verify=True):
    """
    Internal function that calls the actual providers.
    Args:
        token (str): The token used to register on the online backend such
            as the quantum experience.
        url (str): The url used for online backend such as the quantum
            experience.
        proxies (dict): Proxy configuration for the API, as a dict with
            'urls' and credential keys.
        verify (bool): If False, ignores SSL certificates errors.
    Raises:
        QISKitError: if the provider name is not recognized.
    """
    _registered_names = [p.name for p in _DEFAULT_PROVIDER.providers]
    provider_name = _provider_name_from_url(url)
    if provider_name is not None:
        if provider_name not in _registered_names:
            try:
                provider = IBMQProvider(token, url, proxies, verify)
                _DEFAULT_PROVIDER.add_provider(provider)
            except ConnectionError:
                warnings.warn('%s not registered. No connection established.' % provider_name)
        else:
            if REGISTER_CALLED:
                warnings.warn(
                    "%s already registered. Use unregister('%s') to remove." % (
                        provider_name, provider_name))
    else:
        raise QISKitError('Currently only IBMQ providers can be registered.')


def get_qconfig_credentials():
    """
    Looks for registration information in a Qconfig.py
    file in the cwd.

    Returns:
        bool: Did registration occur.

    Raises:
        QISKitError: Error loading Qconfig.py
    """
    did_register = 0
    cwd = os.getcwd()
    if os.path.isfile(cwd + '/Qconfig.py'):
        try:
            spec = importlib.util.spec_from_file_location(
                "Qconfig", cwd+'/Qconfig.py')
            q_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(q_config)
            cdict = q_config.config
            token = q_config.APItoken
            url = cdict['url']
            hub = cdict['hub'] if 'hub' in cdict.keys() else None
            group = cdict['group'] if 'group' in cdict.keys() else None
            project = cdict['project'] if 'project' in cdict.keys() else None
            url = _parse_ibmq_credentials(url, hub, group, project)
            _register(token, url)
            warnings.warn('Loading credentials from a Qconfig is depreciated.',
                          DeprecationWarning)
            did_register = 1
        except Exception:
            raise QISKitError('Error loading Qconfig.py')
    return did_register


def get_env_credentials():
    """
    Looks for registration information in the environment variables.

    Returns:
        bool: Did registration occur.

    Raises:
        QISKitError: Specific provider not found.
    """
    did_register = 0
    token = os.environ.get('QE_TOKEN')
    url = os.environ.get('QE_URL')
    hub = os.environ.get('QE_HUB')
    group = os.environ.get('QE_GROUP')
    project = os.environ.get('QE_PROJECT')

    if token is not None:
        # We have at least a token so lets load it
        if url is not None:
            url = _parse_ibmq_credentials(url, hub, group, project)
        _register(token=token, url=url)
        did_register = 1
    return did_register
