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

import importlib.metadata
import importlib.util
import os
import sys
import warnings

try:
    importlib.metadata.version("qiskit-terra")
except importlib.metadata.PackageNotFoundError:
    # All good!
    pass
else:
    # 'qiskit.tools' is present in all 0.x series of Qiskit and not in Qiskit 1.0+.  If a dev has an
    # editable install and switches from 0.x branches to 1.0+ branches, they might have an empty
    # `qiskit/tools` folder in their path, which will appear as a "namespace package" with no valid
    # location.  We catch that case as "not actually having Qiskit 0.x" as a convenience to devs.
    _has_tools = getattr(importlib.util.find_spec("qiskit.tools"), "has_location", False)
    _suppress_error = os.environ.get("QISKIT_SUPPRESS_1_0_IMPORT_ERROR", False) == "1"
    if not _suppress_error and _has_tools:
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
if sys.version_info < (3, 9):
    warnings.warn(
        "Using Qiskit with Python 3.8 is deprecated as of the 1.1.0 release. "
        "Support for running Qiskit with Python 3.8 will be removed in the "
        "1.3.0 release, which coincides with when Python 3.8 goes end of life.",
        DeprecationWarning,
    )


import qiskit._accelerate
import qiskit._numpy_compat

# Globally define compiled submodules. The normal import mechanism will not find compiled submodules
# in _accelerate because it relies on file paths, but PyO3 generates only one shared library file.
# We manually define them on import so people can directly import qiskit._accelerate.* submodules
# and not have to rely on attribute access.  No action needed for top-level extension packages.
sys.modules["qiskit._accelerate.circuit"] = qiskit._accelerate.circuit
sys.modules["qiskit._accelerate.convert_2q_block_matrix"] = (
    qiskit._accelerate.convert_2q_block_matrix
)
sys.modules["qiskit._accelerate.dense_layout"] = qiskit._accelerate.dense_layout
sys.modules["qiskit._accelerate.error_map"] = qiskit._accelerate.error_map
sys.modules["qiskit._accelerate.isometry"] = qiskit._accelerate.isometry
sys.modules["qiskit._accelerate.uc_gate"] = qiskit._accelerate.uc_gate
sys.modules["qiskit._accelerate.euler_one_qubit_decomposer"] = (
    qiskit._accelerate.euler_one_qubit_decomposer
)
sys.modules["qiskit._accelerate.nlayout"] = qiskit._accelerate.nlayout
sys.modules["qiskit._accelerate.optimize_1q_gates"] = qiskit._accelerate.optimize_1q_gates
sys.modules["qiskit._accelerate.pauli_expval"] = qiskit._accelerate.pauli_expval
sys.modules["qiskit._accelerate.qasm2"] = qiskit._accelerate.qasm2
sys.modules["qiskit._accelerate.qasm3"] = qiskit._accelerate.qasm3
sys.modules["qiskit._accelerate.results"] = qiskit._accelerate.results
sys.modules["qiskit._accelerate.sabre"] = qiskit._accelerate.sabre
sys.modules["qiskit._accelerate.sampled_exp_val"] = qiskit._accelerate.sampled_exp_val
sys.modules["qiskit._accelerate.sparse_pauli_op"] = qiskit._accelerate.sparse_pauli_op
sys.modules["qiskit._accelerate.stochastic_swap"] = qiskit._accelerate.stochastic_swap
sys.modules["qiskit._accelerate.two_qubit_decompose"] = qiskit._accelerate.two_qubit_decompose
sys.modules["qiskit._accelerate.vf2_layout"] = qiskit._accelerate.vf2_layout

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

_config = _user_config.get_config()

from qiskit.compiler import transpile, assemble, schedule, sequence

from .version import __version__

__all__ = [
    "AncillaRegister",
    "ClassicalRegister",
    "MissingOptionalLibraryError",
    "QiskitError",
    "QuantumCircuit",
    "QuantumRegister",
    "assemble",
    "schedule",
    "sequence",
    "transpile",
]
