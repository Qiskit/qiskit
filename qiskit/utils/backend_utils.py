# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""backend utility functions"""

import logging
from qiskit.utils.deprecation import deprecate_func

logger = logging.getLogger(__name__)

_UNSUPPORTED_BACKENDS = ["unitary_simulator", "clifford_simulator"]

# pylint: disable=no-name-in-module,unused-import


class ProviderCheck:
    """Contains Provider verification info."""

    def __init__(self) -> None:
        self.has_ibmq = False
        self.checked_ibmq = False
        self.has_aer = False
        self.checked_aer = False


_PROVIDER_CHECK = ProviderCheck()


def _get_backend_interface_version(backend):
    """Get the backend version int."""
    backend_interface_version = getattr(backend, "version", None)
    return backend_interface_version


def _get_backend_provider(backend):
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version > 1:
        provider = backend.provider
    else:
        provider = backend.provider()
    return provider


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def has_ibmq():
    """Check if IBMQ is installed."""
    if not _PROVIDER_CHECK.checked_ibmq:
        try:
            from qiskit.providers.ibmq import IBMQFactory
            from qiskit.providers.ibmq.accountprovider import AccountProvider

            _PROVIDER_CHECK.has_ibmq = True
        except Exception as ex:  # pylint: disable=broad-except
            _PROVIDER_CHECK.has_ibmq = False
            logger.debug("IBMQFactory/AccountProvider not loaded: '%s'", str(ex))

        _PROVIDER_CHECK.checked_ibmq = True

    return _PROVIDER_CHECK.has_ibmq


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def has_aer():
    """Check if Aer is installed."""
    if not _PROVIDER_CHECK.checked_aer:
        try:
            from qiskit.providers.aer import AerProvider

            _PROVIDER_CHECK.has_aer = True
        except Exception as ex:  # pylint: disable=broad-except
            _PROVIDER_CHECK.has_aer = False
            logger.debug("AerProvider not loaded: '%s'", str(ex))

        _PROVIDER_CHECK.checked_aer = True

    return _PROVIDER_CHECK.has_aer


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_aer_provider(backend):
    """Detect whether or not backend is from Aer provider.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is AerProvider
    """
    if has_aer():
        from qiskit.providers.aer import AerProvider

        if isinstance(_get_backend_provider(backend), AerProvider):
            return True
        from qiskit.providers.aer.backends.aerbackend import AerBackend

        return isinstance(backend, AerBackend)

    return False


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_basicaer_provider(backend):
    """Detect whether or not backend is from BasicAer provider.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is BasicAer
    """
    from qiskit.providers.basicaer import BasicAerProvider

    return isinstance(_get_backend_provider(backend), BasicAerProvider)


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_ibmq_provider(backend):
    """Detect whether or not backend is from IBMQ provider.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is IBMQ
    """
    if has_ibmq():
        from qiskit.providers.ibmq.accountprovider import AccountProvider

        return isinstance(_get_backend_provider(backend), AccountProvider)

    return False


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_aer_statevector_backend(backend):
    """
    Return True if backend object is statevector and from Aer provider.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is statevector
    """
    return is_statevector_backend(backend) and is_aer_provider(backend)


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_statevector_backend(backend):
    """
    Return True if backend object is statevector.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is statevector
    """
    if has_aer():
        from qiskit.providers.aer.backends import AerSimulator, StatevectorSimulator

        if isinstance(backend, StatevectorSimulator):
            return True
        if isinstance(backend, AerSimulator) and "aer_simulator_statevector" in backend.name():
            return True
    if backend is None:
        return False
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version <= 1:
        return backend.name().startswith("statevector")
    else:
        return backend.name.startswith("statevector")


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_simulator_backend(backend):
    """
    Return True if backend is a simulator.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is a simulator
    """
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version <= 1:
        return backend.configuration().simulator
    return False


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_local_backend(backend):
    """
    Return True if backend is a local backend.

    Args:
        backend (Backend): backend instance
    Returns:
        bool: True is a local backend
    """
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version <= 1:
        return backend.configuration().local
    return False


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def is_aer_qasm(backend):
    """
    Return True if backend is Aer Qasm simulator
    Args:
        backend (Backend): backend instance

    Returns:
        bool: True is Aer Qasm simulator
    """
    ret = False
    if is_aer_provider(backend):
        if not is_statevector_backend(backend):
            ret = True
    return ret


@deprecate_func(
    since="0.24.0",
    additional_msg="For code migration guidelines, visit https://qisk.it/qi_migration.",
)
def support_backend_options(backend):
    """
    Return True if backend supports backend_options
    Args:
        backend (Backend): backend instance

    Returns:
        bool: True is support backend_options
    """
    ret = False
    if is_basicaer_provider(backend) or is_aer_provider(backend):
        ret = True
    return ret
