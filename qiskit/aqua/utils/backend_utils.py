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

""" backend utility functions """

from collections import OrderedDict
import importlib
import logging
from qiskit.aqua import Preferences

logger = logging.getLogger(__name__)

_UNSUPPORTED_BACKENDS = ['unitary_simulator', 'clifford_simulator']

# pylint: disable=no-name-in-module, import-error, unused-import


class ProviderCheck:
    """Contains Provider verification info."""

    def __init__(self) -> None:
        self.has_ibmq = False
        self.checked_ibmq = False
        self.has_aer = False
        self.checked_aer = False


_PROVIDER_CHECK = ProviderCheck()


def has_ibmq():
    """ Check if IBMQ is installed """
    if not _PROVIDER_CHECK.checked_ibmq:
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit.providers.ibmq import IBMQFactory
            # pylint: disable=syntax-error
            from qiskit.providers.ibmq.accountprovider import AccountProvider
            _PROVIDER_CHECK.has_ibmq = True
        except Exception as ex:  # pylint: disable=broad-except
            _PROVIDER_CHECK.has_ibmq = False
            logger.debug("IBMQFactory/AccountProvider not loaded: '%s'", str(ex))

        _PROVIDER_CHECK.checked_ibmq = True

    return _PROVIDER_CHECK.has_ibmq


def has_aer():
    """ check if Aer is installed """
    if not _PROVIDER_CHECK.checked_aer:
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit.providers.aer import AerProvider
            _PROVIDER_CHECK.has_aer = True
        except Exception as ex:  # pylint: disable=broad-except
            _PROVIDER_CHECK.has_aer = False
            logger.debug("AerProvider not loaded: '%s'", str(ex))

        _PROVIDER_CHECK.checked_aer = True

    return _PROVIDER_CHECK.has_aer


def is_aer_provider(backend):
    """Detect whether or not backend is from Aer provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is AerProvider
    """
    if has_aer():
        # pylint: disable=import-outside-toplevel
        from qiskit.providers.aer import AerProvider
        return isinstance(backend.provider(), AerProvider)

    return False


def is_basicaer_provider(backend):
    """Detect whether or not backend is from BasicAer provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is BasicAer
    """
    # pylint: disable=import-outside-toplevel
    from qiskit.providers.basicaer import BasicAerProvider

    return isinstance(backend.provider(), BasicAerProvider)


def is_ibmq_provider(backend):
    """Detect whether or not backend is from IBMQ provider.

    Args:
        backend (BaseBackend): backend instance
    Returns:
        bool: True is IBMQ
    """
    if has_ibmq():
        # pylint: disable=syntax-error,import-outside-toplevel
        from qiskit.providers.ibmq.accountprovider import AccountProvider
        return isinstance(backend.provider(), AccountProvider)

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
    """ returns Aer backend """
    providers = ['qiskit.Aer', 'qiskit.BasicAer']
    for provider in providers:
        try:
            return get_backend_from_provider(provider, backend_name)
        except Exception:  # pylint: disable=broad-except
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
    is_ibmq = False
    if has_ibmq():
        # pylint: disable=import-outside-toplevel
        from qiskit.providers.ibmq import IBMQFactory
        if isinstance(provider_object, IBMQFactory):
            is_ibmq = True
            # enable IBMQ account
            provider = _refresh_ibmq_account()
            if provider is not None:
                return [x.name() for x in provider.backends()
                        if x.name() not in _UNSUPPORTED_BACKENDS]

    if not is_ibmq:
        try:
            # try as variable containing provider instance
            return [x.name() for x in provider_object.backends()
                    if x.name() not in _UNSUPPORTED_BACKENDS]
        except Exception:  # pylint: disable=broad-except
            # try as provider class then
            try:
                provider_instance = provider_object()
                return [x.name() for x in provider_instance.backends()
                        if x.name() not in _UNSUPPORTED_BACKENDS]
            except Exception:  # pylint: disable=broad-except
                pass

    raise ImportError("'Backends not found for provider '{}'".format(provider_name))


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
    provider_object = _load_provider(provider_name)
    is_ibmq = False
    if has_ibmq():
        # pylint: disable=import-outside-toplevel
        from qiskit.providers.ibmq import IBMQFactory
        if isinstance(provider_object, IBMQFactory):
            is_ibmq = True
            # enable IBMQ account
            provider = _refresh_ibmq_account()
            if provider is not None:
                return provider.get_backend(backend_name)

    if not is_ibmq:
        try:
            # try as variable containing provider instance
            return provider_object.get_backend(backend_name)
        except Exception:  # pylint: disable=broad-except
            # try as provider class then
            try:
                provider_instance = provider_object()
                return provider_instance.get_backend(backend_name)
            except Exception:  # pylint: disable=broad-except
                pass

    raise ImportError("'{} not found in provider '{}'".format(backend_name, provider_name))


def get_local_providers():
    """ returns local providers """
    providers = OrderedDict()
    for provider in ['qiskit.Aer', 'qiskit.BasicAer']:
        try:
            providers[provider] = get_backends_from_provider(provider)
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug("'%s' not loaded: '%s'.", provider, str(ex))

    return providers


def register_ibmq_and_get_known_providers():  # pylint: disable=invalid-name
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
    # pylint: disable=import-outside-toplevel
    from qiskit.providers import BaseBackend

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
        except Exception:  # pylint: disable=broad-except
            pass

    raise ImportError(
        "Backend '{}' not found in providers {}".format(backend, list(known_providers.values())))


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


def _refresh_ibmq_account():
    """
    Refresh IBMQ account by enabling or disabling it depending on preferences stored values
    """
    preferences = Preferences().ibmq_credentials_preferences
    token = preferences.token or ''
    proxies = preferences.proxies or {}
    hub = preferences.hub
    group = preferences.group
    project = preferences.project
    provider = None
    try:
        # pylint: disable=no-name-in-module,import-error,import-outside-toplevel
        from qiskit import IBMQ
        providers = IBMQ.providers()
        if token != '':
            # check if there was a previous account that needs to be disabled first
            disable_account = False
            enable_account = True
            for provider in providers:
                if provider.credentials.token == token and provider.credentials.proxies == proxies:
                    enable_account = False
                else:
                    disable_account = True

            if disable_account:
                IBMQ.disable_account()
                logger.info('Disabled IBMQ account.')

            if enable_account:
                IBMQ.enable_account(token, proxies=proxies)
                logger.info('Enabled IBMQ account.')

            providers = IBMQ.providers(hub=hub, group=group, project=project)
            provider = providers[0] if providers else None
            if provider is None:
                logger.info("No Provider found for IBMQ account. "
                            "Hub/Group/Project: '%s/%s/%s' Proxies:'%s'",
                            hub, group, project, proxies)
        else:
            if providers:
                IBMQ.disable_account()
                logger.info('Disabled IBMQ account.')
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("IBMQ account Account Failure. "
                       "Hub/Group/Project: '%s/%s/%s' "
                       "Proxies:'%s' :%s", hub, group, project, proxies, str(ex))

    return provider


def _get_ibmq_provider():
    """Registers IBMQ and return it."""
    providers = OrderedDict()
    try:
        providers['qiskit.IBMQ'] = get_backends_from_provider('qiskit.IBMQ')
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to access IBMQ: %s", str(ex))

    return providers
