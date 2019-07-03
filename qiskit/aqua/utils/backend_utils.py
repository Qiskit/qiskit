# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections import OrderedDict
import importlib
import logging
from qiskit.providers import BaseBackend
from qiskit.providers.basicaer import BasicAerProvider
from qiskit.aqua import Preferences

logger = logging.getLogger(__name__)

try:
    # pylint: disable=no-name-in-module, import-error
    from qiskit.providers.ibmq import IBMQFactory
    from qiskit.providers.ibmq.accountprovider import AccountProvider
    HAS_IBMQ = True
except Exception as ex:
    HAS_IBMQ = False
    logger.debug("IBMQFactory/AccountProvider not loaded: '{}'".format(str(ex)))

try:
    from qiskit.providers.aer import AerProvider
    HAS_AER = True
except Exception as ex:
    HAS_AER = False
    logger.debug("AerProvider not loaded: '{}'".format(str(ex)))

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
        bool: True is AerProvider
    """
    if has_aer():
        return isinstance(backend.provider(), AerProvider)
    else:
        return False


def is_basicaer_provider(backend):
    """Detect whether or not backend is from BasicAer provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is BasicAer
    """
    return isinstance(backend.provider(), BasicAerProvider)


def is_ibmq_provider(backend):
    """Detect whether or not backend is from IBMQ provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is IBMQ
    """
    if has_ibmq():
        return isinstance(backend.provider(), AccountProvider)
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


def is_aer_qasm(backend):
    """
    Return True if backend is Aer Qasm simulator
    Args:
        backend (BaseBackend): backend instance

    Returns:
        bool: True is Aer Qasm simulator
    """
    ret = False
    if is_aer_provider(backend):
        if not is_statevector_backend(backend):
            ret = True
    return ret


def support_backend_options(backend):
    """
    Return True if backend supports backend_options
    Args:
        backend (BaseBackend): backend instance

    Returns:
        bool: True is support backend_options
    """
    ret = False
    if is_basicaer_provider(backend) or is_aer_provider(backend):
        ret = True
    return ret


def get_aer_backend(backend_name):
    providers = ['qiskit.Aer', 'qiskit.BasicAer']
    for provider in providers:
        try:
            return get_backend_from_provider(provider, backend_name)
        except Exception:
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
    if has_ibmq() and isinstance(provider_object, IBMQFactory):
        # enable IBMQ account
        preferences = Preferences().ibmq_credentials_preferences
        provider = _enable_ibmq_account(preferences.url,
                                        preferences.token,
                                        preferences.proxies,
                                        preferences.hub,
                                        preferences.group,
                                        preferences.project)
        if provider is not None:
            provider_object = provider

        return [x.name() for x in provider_object.backends() if x.name() not in _UNSUPPORTED_BACKENDS]

    try:
        # try as variable containing provider instance
        return [x.name() for x in provider_object.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
    except Exception:
        # try as provider class then
        try:
            provider_instance = provider_object()
            return [x.name() for x in provider_instance.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
        except Exception:
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
    if has_ibmq() and isinstance(provider_object, IBMQFactory):
        preferences = Preferences().ibmq_credentials_preferences
        provider = _enable_ibmq_account(preferences.url,
                                        preferences.token,
                                        preferences.proxies,
                                        preferences.hub,
                                        preferences.group,
                                        preferences.project)
        if provider is not None:
            provider_object = provider

        backend = provider_object.get_backend(backend_name)
    else:
        try:
            # try as variable containing provider instance
            backend = provider_object.get_backend(backend_name)
        except Exception:
            # try as provider class then
            try:
                provider_instance = provider_object()
                backend = provider_instance.get_backend(backend_name)
            except Exception:
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


def get_provider_from_backend(backend):
    """
    Attempts to find a known provider that provides this backend.

    Args:
        backend (BaseBackend or str): backend object or backend name
    Returns:
        str: provider name
    Raises:
        ImportError: Failed to find provider
    """
    known_providers = {
                       'BasicAerProvider': 'qiskit.BasicAer',
                       'AerProvider': 'qiskit.Aer',
                       'IBMQFactory': 'qiskit.IBMQ',
                       }
    if isinstance(backend, BaseBackend):
        provider = backend.provider()
        if provider is None:
            raise ImportError("Backend object '{}' has no provider".format(backend.name()))

        return known_providers.get(provider.__class__.__name__, provider.__class__.__qualname__)
    elif not isinstance(backend, str):
        raise ImportError("Invalid Backend '{}'".format(backend))

    for provider in known_providers.values():
        try:
            if get_backend_from_provider(provider, backend) is not None:
                return provider
        except Exception:
            pass

    raise ImportError("Backend '{}' not found in providers {}".format(backend, list(known_providers.values())))


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

    return provider_object


def _enable_ibmq_account(url, token, proxies, hub, group, project):
    """
    Enable IBMQ account, if not alreay enabled.
    """
    provider = None
    if not has_ibmq():
        return provider
    try:
        url = url or ''
        token = token or ''
        proxies = proxies or {}
        if url != '' and token != '':
            # pylint: disable=no-name-in-module, import-error
            from qiskit import IBMQ
            if IBMQ._v1_provider._accounts:
                from qiskit.providers.ibmq.credentials import Credentials
                credentials = Credentials(token, url, proxies=proxies)
                unique_id = credentials.unique_id()
                if unique_id in IBMQ._v1_provider._accounts:
                    # disable first any existent previous account with same unique_id and different properties
                    enabled_credentials = IBMQ._v1_provider._accounts[unique_id].credentials
                    if enabled_credentials.url != url or enabled_credentials.token != token or enabled_credentials.proxies != proxies:
                        del IBMQ._v1_provider._accounts[unique_id]
                    else:
                        return IBMQ._v1_provider
            elif IBMQ._credentials:
                enabled_credentials = IBMQ._credentials
                # disable first any existent previous account with same unique_id and different properties
                if enabled_credentials.url != url or enabled_credentials.token != token or enabled_credentials.proxies != proxies:
                    IBMQ.disable_account()
                else:
                    providers = IBMQ.providers(hub=hub, group=group, project=project)
                    return providers[0] if providers else None

            IBMQ.enable_account(token, url=url, proxies=proxies)
            providers = IBMQ.providers(hub=hub, group=group, project=project)
            provider = providers[0] if providers else None
            if provider is not None:
                logger.info("Enabled IBMQ account. Token:'{}' Url:'{}' "
                            "H/G/P: '{}/{}/{}' Proxies:'{}'".format(token, url, hub, group, project, proxies))
    except Exception as ex:
        logger.warning("Failed to enable IBMQ account. Token:'{}' Url:'{}' "
                       "H/G/P: '{}/{}/{}' Proxies:'{}' :{}".format(token, url, hub, group, project, proxies, str(ex)))

    return provider


def disable_ibmq_account(url, token, proxies):
    """Disable IBMQ account."""
    if not has_ibmq():
        return
    try:
        url = url or ''
        token = token or ''
        proxies = proxies or {}
        if url != '' and token != '':
            # pylint: disable=no-name-in-module, import-error
            from qiskit import IBMQ
            if IBMQ._v1_provider._accounts:
                from qiskit.providers.ibmq.credentials import Credentials
                credentials = Credentials(token, url, proxies=proxies)
                unique_id = credentials.unique_id()
                if unique_id in IBMQ._v1_provider._accounts:
                    del IBMQ._v1_provider._accounts[unique_id]
                    logger.info("Disabled IBMQ v1 account. Url:'{}' "
                                "Token:'{}' Proxies:'{}'".format(url, token, proxies))
                else:
                    logger.info("IBMQ v1 account is not active. Not disabled. "
                                "Url:'{}' Token:'{}' Proxies:'{}'".format(url, token, proxies))
            elif IBMQ._credentials:
                enabled_credentials = IBMQ._credentials
                if enabled_credentials.url == url and enabled_credentials.token == token and enabled_credentials.proxies == proxies:
                    IBMQ.disable_account()
                else:
                    logger.info("IBMQ v2 account is not active. Not disabled. "
                                "Token:'{}' Url:'{}' Proxies:'{}'".format(token, url, proxies))
    except Exception as ex:
        logger.warning("Failed to disable IBMQ account. Token:'{}' "
                       "Url:'{}' Proxies:'{}' :{}".format(token, url, proxies, str(ex)))


def _get_ibmq_provider():
    """Registers IBMQ and return it."""
    providers = OrderedDict()
    try:
        providers['qiskit.IBMQ'] = get_backends_from_provider('qiskit.IBMQ')
    except Exception as ex:
        logger.warning("Failed to access IBMQ: {}".format(str(ex)))

    return providers
