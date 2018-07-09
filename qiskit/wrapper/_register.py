# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Internal register function"""

import warnings
from qiskit import QISKitError
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from qiskit.wrapper.credentials import _settings as rc_set

# Default provider used by the rest of the functions on this module. Please
# note that this is a global object.
_DEFAULT_PROVIDER = DefaultQISKitProvider()


def _register(token=None, url='https://quantumexperience.ng.bluemix.net/api',
              hub=None, group=None, project=None, proxies=None, verify=True):
    """
    Internal function that calls the actual providers.
    Args:
        token (str): The token used to register on the online backend such
            as the quantum experience.
        url (str): The url used for online backend such as the quantum
            experience.
        hub (str): The hub used for online backend.
        group (str): The group used for online backend.
        project (str): The project used for online backend.
        proxies (dict): Proxy configuration for the API, as a dict with
            'urls' and credential keys.
        verify (bool): If False, ignores SSL certificates errors.
    Raises:
        QISKitError: if the provider name is not recognized.
    """
    _registered_names = [p.name for p in _DEFAULT_PROVIDER.providers]
    if 'quantumexperience' in url:
        provider_name = 'ibmq'
    elif 'q-console' in url:
        provider_name = 'qnet'
    else:
        raise QISKitError('Unkown provider name.')
    if provider_name not in _registered_names:
        try:
            provider = IBMQProvider(token, url,
                                    hub, group, project,
                                    proxies, verify)
            _DEFAULT_PROVIDER.add_provider(provider)
        except ConnectionError:
            warnings.warn('%s not registered. No connection established.' % provider_name)
    else:
        if rc_set.REGISTER_CALLED:
            warnings.warn(
                "%s already registered. Use unregister('%s') to remove." % (
                    provider_name, provider_name))
