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

import os
import pkgutil
import sys
import warnings
import importlib.metadata

try:
    _qiskit_version = importlib.metadata.version("qiskit")
except importlib.metadata.PackageNotFoundError:
    warnings.warn(
        "The `qiskit` package is not installed, only `qiskit-terra` is installed."
        " Starting in Qiskit 1.0.0 only the `qiskit` package will be published."
        " Migrate any requirements files still using `qiskit-terra` to use `qiskit` instead."
        " See https://qisk.it/1-0-packaging-migration for more detail.",
        FutureWarning,
        stacklevel=2,
    )
else:
    _major, _ = _qiskit_version.split(".", 1)
    _suppress_error = os.environ.get("QISKIT_SUPPRESS_1_0_IMPORT_ERROR", False) == "1"
    if int(_major) > 0 and not _suppress_error:
        raise ImportError(
            "Qiskit is installed in an invalid environment that has both Qiskit >=1.0"
            " and an earlier version."
            " You should create a new virtual environment, and ensure that you do not mix"
            " dependencies between Qiskit <1.0 and >=1.0."
            " Any packages that depend on 'qiskit-terra' are not compatible with Qiskit 1.0 and"
            " will need to be updated."
            " Qiskit unfortunately cannot enforce this requirement during environment resolution."
            " See https://qisk.it/packaging-1-0 for more detail."
        )

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
sys.modules[
    "qiskit._accelerate.convert_2q_block_matrix"
] = qiskit._accelerate.convert_2q_block_matrix


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

import qiskit.circuit.measure
import qiskit.circuit.reset

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code AND *before*
# importing the package you want to allow extensions for (in this case `backends`).

# Support for the deprecated extending this namespace.
# Remove this after 0.46.0 release
__path__ = pkgutil.extend_path(__path__, __name__)

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
            warnings.warn(
                "The 'qiskit.Aer' entry point is deprecated and will be removed in Qiskit 1.0."
                " You should use 'qiskit_aer.Aer' directly instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                import qiskit_aer

                self.aer = qiskit_aer.Aer
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.aer:
            warnings.warn(
                "The 'qiskit.Aer' entry point is deprecated and will be removed in Qiskit 1.0."
                " You should use 'qiskit_aer.Aer' directly instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                import qiskit_aer

                self.aer = qiskit_aer.Aer
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
            warnings.warn(
                "The qiskit.IBMQ entrypoint and the qiskit-ibmq-provider package ("
                "accessible from 'qiskit.providers.ibmq`) are deprecated and will be removed "
                "in a future release. Instead you should use the qiskit-ibm-provider package "
                "which is accessible from 'qiskit_ibm_provider'. You can install it with "
                "'pip install qiskit_ibm_provider'",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                from qiskit.providers import ibmq

                self.ibmq = ibmq.IBMQ
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.ibmq:
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
            try:
                from qiskit.providers import ibmq

                self.ibmq = ibmq.IBMQ
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

# lazily deprecate BasicAer import (global instance)
_DEPRECATED_NAMES = {
    "BasicAer": "qiskit.providers.basicaer",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        module_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"{name} is deprecated since Qiskit 0.46 and will be removed in Qiskit 1.0. "
            f"The BasicAer (qiskit.providers.basicaer) module has been superseded by  "
            f"qiskit.providers.basic_provider, and all its classes have been renamed "
            f"to follow a new naming convention. More information and migration guidelines "
            f"can be found in the 0.46 API docs for BasicAer.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_name), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
