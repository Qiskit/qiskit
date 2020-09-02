# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" backend utility functions """

import logging

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
