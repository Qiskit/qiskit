# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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

from collections import OrderedDict
import importlib
import logging

try:
    from qiskit.providers.ibmq import IBMQProvider
    HAS_IBMQ = True
except ImportError:
    HAS_IBMQ = False
    pass
try:
    from qiskit.providers.aer import AerProvider
    HAS_AER = True
except ImportError:
    HAS_AER = False
    pass

logger = logging.getLogger(__name__)


_UNSUPPORTED_BACKENDS = ['unitary_simulator', 'clifford_simulator']


def has_ibmq():
    return HAS_IBMQ


def has_aer():
    return HAS_AER


def is_aer_provider(backend):
    """Detect whether or not backend is from Aer provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is statevector
    """
    if has_aer():
        return isinstance(backend.provider(), AerProvider)
    else:
        return False


def is_aer_statevector_backend(backend):
    """
    Return True if backend object is statevector and from Aer provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is statevector
    """
    return is_statevector_backend(backend) and is_aer_provider(backend)


def is_statevector_backend(backend):
    """
    Return True if backend object is statevector.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is statevector
    """
    return backend.name().startswith('statevector') if backend is not None else False


def is_simulator_backend(backend):
    """
    Return True if backend is a simulator.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is a simulator
    """
    return backend.configuration().simulator


def is_local_backend(backend):
    """
    Return True if backend is a local backend.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is a local backend
    """
    return backend.configuration().local


def is_ibmq_provider(backend):
    """Detect whether or not backend is from IBMQ provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is statevector
    """
    if has_ibmq():
        return isinstance(backend.provider(), IBMQProvider)
    else:
        return False


def get_aer_backend(backend_name):
    providers = ['qiskit.Aer', 'qiskit.BasicAer']
    for provider in providers:
        try:
            return get_backend_from_provider(provider, backend_name)
        except:
            pass

    raise ImportError("Backend '{}' not found in providers {}".format(backend_name, providers))


def get_backends_from_provider(provider_name):
    """
    Backends access method.

    Args:
        provider_name (str): Fullname of provider instance global property or class
    Returns:
        list: backend names
    Raises:
        ImportError: Invalid provider name or failed to find provider
    """
    provider_object = _load_provider(provider_name)
    if has_ibmq() and isinstance(provider_object, IBMQProvider):
        from qiskit_aqua_cmd import Preferences
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        kwargs = {}
        if url is not None and url != '':
            kwargs['url'] = url
        if token is not None and token != '':
            kwargs['token'] = token
        return [x.name() for x in provider_object.backends(**kwargs) if x.name() not in _UNSUPPORTED_BACKENDS]

    try:
        # try as variable containing provider instance
        return [x.name() for x in provider_object.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
    except:
        # try as provider class then
        try:
            provider_instance = provider_object()
            return [x.name() for x in provider_instance.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
        except:
            pass

    raise ImportError("'Backends not found for provider '{}'".format(provider_object))


def get_backend_from_provider(provider_name, backend_name):
    """
    Backend access method.

    Args:
        provider_name (str): Fullname of provider instance global property or class
        backend_name (str): name of backend for this provider
    Returns:
        BaseBackend: backend object
    Raises:
        ImportError: Invalid provider name or failed to find provider
    """
    backend = None
    provider_object = _load_provider(provider_name)
    if has_ibmq() and isinstance(provider_object, IBMQProvider):
        from qiskit_aqua_cmd import Preferences
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        kwargs = {}
        if url is not None and url != '':
            kwargs['url'] = url
        if token is not None and token != '':
            kwargs['token'] = token
        backend = provider_object.get_backend(backend_name, **kwargs)
    else:
        try:
            # try as variable containing provider instance
            backend = provider_object.get_backend(backend_name)
        except:
            # try as provider class then
            try:
                provider_instance = provider_object()
                backend = provider_instance.get_backend(backend_name)
            except:
                pass

    if backend is None:
        raise ImportError("'{} not found in provider '{}'".format(backend_name, provider_object))

    return backend


def get_local_providers():
    providers = OrderedDict()
    for provider in ['qiskit.Aer', 'qiskit.BasicAer']:
        try:
            providers[provider] = get_backends_from_provider(provider)
        except Exception as e:
            logger.debug("'{}' not loaded: '{}'.".format(provider, str(e)))

    return providers


def register_ibmq_and_get_known_providers():
    """Gets known local providers and registers IBMQ."""
    providers = get_local_providers()
    if has_ibmq():
        providers.update(_get_ibmq_provider())
    return providers


def get_provider_from_backend(backend_name):
    """
    Attempts to find a known provider that provides this backend.

    Args:
        backend_name (str): name of backend for tgis provider
    Returns:
        str: provider name
    Raises:
        ImportError: Failed to find provider
    """
    providers = ['qiskit.Aer', 'qiskit.BasicAer', 'qiskit.IBMQ']
    for provider in providers:
        try:
            if get_backend_from_provider(provider, backend_name) is not None:
                return provider
        except:
            pass

    raise ImportError("Backend '{}' not found in providers {}".format(backend_name, providers))


def _load_provider(provider_name):
    index = provider_name.rfind(".")
    if index < 1:
        raise ImportError("Invalid provider name '{}'".format(provider_name))

    modulename = provider_name[0:index]
    objectname = provider_name[index + 1:len(provider_name)]

    module = importlib.import_module(modulename)
    if module is None:
        raise ImportError("Failed to import provider '{}'".format(provider_name))

    provider_object = getattr(module, objectname)
    if provider_object is None:
        raise ImportError("Failed to import provider '{}'".format(provider_name))

    if has_ibmq() and isinstance(provider_object, IBMQProvider):
        # enable IBMQ account
        from qiskit_aqua_cmd import Preferences
        preferences = Preferences()
        enable_ibmq_account(preferences.get_url(), preferences.get_token(), preferences.get_proxies({}))

    return provider_object


def enable_ibmq_account(url, token, proxies):
    """
    Enable IBMQ account, if not alreay enabled.
    """
    if not has_ibmq():
        return
    try:
        url = url or ''
        token = token or ''
        proxies = proxies or {}
        if url != '' and token != '':
            from qiskit import IBMQ
            from qiskit.providers.ibmq.credentials import Credentials
            credentials = Credentials(token, url, proxies=proxies)
            unique_id = credentials.unique_id()
            if unique_id in IBMQ._accounts:
                # disable first any existent previous account with same unique_id and different properties
                enabled_credentials = IBMQ._accounts[unique_id].credentials
                if enabled_credentials.url != url or enabled_credentials.token != token or enabled_credentials.proxies != proxies:
                    del IBMQ._accounts[unique_id]

            if unique_id not in IBMQ._accounts:
                IBMQ.enable_account(token, url=url, proxies=proxies)
                logger.info("Enabled IBMQ account. Url:'{}' Token:'{}' "
                            "Proxies:'{}'".format(url, token, proxies))
    except Exception as e:
        logger.warning("Failed to enable IBMQ account. Url:'{}' Token:'{}' "
                       "Proxies:'{}' :{}".format(url, token, proxies, str(e)))


def disable_ibmq_account(url, token, proxies):
    """Disable IBMQ account."""
    if not has_ibmq():
        return
    try:
        url = url or ''
        token = token or ''
        proxies = proxies or {}
        if url != '' and token != '':
            from qiskit import IBMQ
            from qiskit.providers.ibmq.credentials import Credentials
            credentials = Credentials(token, url, proxies=proxies)
            unique_id = credentials.unique_id()
            if unique_id in IBMQ._accounts:
                del IBMQ._accounts[unique_id]
                logger.info("Disabled IBMQ account. Url:'{}' "
                            "Token:'{}' Proxies:'{}'".format(url, token, proxies))
            else:
                logger.info("IBMQ account is not active. Not disabled. "
                            "Url:'{}' Token:'{}' Proxies:'{}'".format(url, token, proxies))
    except Exception as e:
        logger.warning("Failed to disable IBMQ account. Url:'{}' "
                       "Token:'{}' Proxies:'{}' :{}".format(url, token, proxies, str(e)))


def _get_ibmq_provider():
    """Registers IBMQ and return it."""
    providers = OrderedDict()
    try:
        providers['qiskit.IBMQ'] = get_backends_from_provider('qiskit.IBMQ')
    except Exception as e:
        logger.warning("Failed to access IBMQ: {}".format(str(e)))

    return providers
