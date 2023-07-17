# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=wrong-import-position

"""Main Qiskit public functionality."""

import pkgutil
import sys
import warnings

import qiskit._accelerate


# Globally define compiled submodules. The normal import mechanism will not find compiled submodules
# in _accelerate because it relies on file paths, but PyO3 generates only one shared library file.
# We manually define them on import so people can directly import qiskit._accelerate.* submodules
# and not have to rely on attribute access.  No action needed for top-level extension packages.
sys.modules["qiskit._accelerate.nlayout"] = qiskit._accelerate.nlayout
sys.modules["qiskit._accelerate.stochastic_swap"] = qiskit._accelerate.stochastic_swap
sys.modules["qiskit._accelerate.sabre_swap"] = qiskit._accelerate.sabre_swap
sys.modules["qiskit._accelerate.sabre_layout"] = qiskit._accelerate.sabre_layout
sys.modules["qiskit._accelerate.pauli_expval"] = qiskit._accelerate.pauli_expval
sys.modules["qiskit._accelerate.dense_layout"] = qiskit._accelerate.dense_layout
sys.modules["qiskit._accelerate.sparse_pauli_op"] = qiskit._accelerate.sparse_pauli_op
sys.modules["qiskit._accelerate.results"] = qiskit._accelerate.results
sys.modules["qiskit._accelerate.optimize_1q_gates"] = qiskit._accelerate.optimize_1q_gates
sys.modules["qiskit._accelerate.sampled_exp_val"] = qiskit._accelerate.sampled_exp_val
sys.modules["qiskit._accelerate.vf2_layout"] = qiskit._accelerate.vf2_layout
sys.modules["qiskit._accelerate.error_map"] = qiskit._accelerate.error_map
sys.modules[
    "qiskit._accelerate.euler_one_qubit_decomposer"
] = qiskit._accelerate.euler_one_qubit_decomposer


# Extend namespace for backwards compat
from qiskit import namespace

# Add hook to redirect imports from qiskit.providers.aer* to qiskit_aer*
# this is necessary for backwards compatibility for users when qiskit-aer
# and qiskit-terra shared the qiskit namespace
new_meta_path_finder = namespace.QiskitElementImport("qiskit.providers.aer", "qiskit_aer")
sys.meta_path = [new_meta_path_finder] + sys.meta_path

# qiskit errors operator
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit import QuantumCircuit

# user config
from qiskit import user_config as _user_config

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions
import qiskit.circuit.measure
import qiskit.circuit.reset

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code AND *before*
# importing the package you want to allow extensions for (in this case `backends`).

# Support for the deprecated extending this namespace.
# Remove this after 0.46.0 release
__path__ = pkgutil.extend_path(__path__, __name__)

# Please note these are global instances, not modules.
from qiskit.providers.basicaer import BasicAer

_config = _user_config.get_config()

# Moved to after IBMQ and Aer imports due to import issues
# with other modules that check for IBMQ (tools)
from qiskit.execute_function import execute
from qiskit.compiler import transpile, assemble, schedule, sequence

from .version import __version__
from .version import QiskitVersion


__qiskit_version__ = QiskitVersion()


class AerWrapper:
    """Lazy loading wrapper for Aer provider."""

    def __init__(self):
        self.aer = None

    def __bool__(self):
        if self.aer is None:
            try:
                from qiskit.providers import aer

                self.aer = aer.Aer
                warnings.warn(
                    "The qiskit.Aer entry point will be deprecated in a future release and "
                    "subsequently removed. Instead you should use this "
                    "directly from the root of the qiskit-aer package.",
                    PendingDeprecationWarning,
                    stacklevel=2,
                )
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.aer:
            try:
                from qiskit.providers import aer

                self.aer = aer.Aer
                warnings.warn(
                    "The qiskit.Aer entry point will be deprecated in a future release and "
                    "subsequently removed. Instead you should use this "
                    "directly from the root of the qiskit-aer package.",
                    PendingDeprecationWarning,
                    stacklevel=2,
                )
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    "qiskit-aer", "Aer provider", "pip install qiskit-aer"
                ) from ex
        return getattr(self.aer, attr)


class IBMQWrapper:
    """Lazy loading wrapper for IBMQ provider."""

    def __init__(self):
        self.ibmq = None

    def __bool__(self):
        if self.ibmq is None:
            try:
                from qiskit.providers import ibmq

                self.ibmq = ibmq.IBMQ
                warnings.warn(
                    "The qiskit.IBMQ entrypoint and the qiskit-ibmq-provider package ("
                    "accessible from 'qiskit.providers.ibmq`) are deprecated and will be removed "
                    "in a future release. Instead you should use the qiskit-ibm-provider package "
                    "which is accessible from 'qiskit_ibm_provider'. You can install it with "
                    "'pip install qiskit_ibm_provider'",
                    DeprecationWarning,
                    stacklevel=2,
                )

            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.ibmq:
            try:
                from qiskit.providers import ibmq

                self.ibmq = ibmq.IBMQ
                warnings.warn(
                    "The qiskit.IBMQ entrypoint and the qiskit-ibmq-provider package ("
                    "accessible from 'qiskit.providers.ibmq`) are deprecated and will be removed "
                    "in a future release. Instead you should use the qiskit-ibm-provider package "
                    "which is accessible from 'qiskit_ibm_provider'. You can install it with "
                    "'pip install qiskit_ibm_provider'. Just replace 'qiskit.IBMQ' with "
                    "'qiskit_ibm_provider.IBMProvider'",
                    DeprecationWarning,
                    stacklevel=2,
                )
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    "qiskit-ibmq-provider", "IBMQ provider", "pip install qiskit-ibmq-provider"
                ) from ex
        return getattr(self.ibmq, attr)


Aer = AerWrapper()
IBMQ = IBMQWrapper()

__all__ = [
    "Aer",
    "AncillaRegister",
    "BasicAer",
    "ClassicalRegister",
    "IBMQ",
    "MissingOptionalLibraryError",
    "QiskitError",
    "QuantumCircuit",
    "QuantumRegister",
    "assemble",
    "execute",
    "schedule",
    "sequence",
    "transpile",
]
