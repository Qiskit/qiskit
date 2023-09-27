# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Initialize qubit registers to desired arbitrary state."""

# pylint: disable=wrong-import-position

import os
import platform
import warnings

import numpy as np

# The PyPI-distributed versions of Numpy 1.25 and 1.26 were compiled for macOS x86_64 using a
# compiler that caused a bug in the complex-multiply ufunc when AVX2 extensions are enabled.  This
# severely affects the `Isometry` definition code to the point of the returned definitions being
# entirely unsound, so we need to warn users.  See:
#
# - https://github.com/Qiskit/qiskit/issues/10305
# - https://github.com/numpy/numpy/issues/24000
_KNOWN_AFFECTED_NUMPY_VERSIONS = ("1.25.0", "1.25.1", "1.25.2", "1.26.0")
_IS_BAD_NUMPY = (
    os.environ.get("QISKIT_CMUL_AVX2_GOOD_NUMPY", "0") != "1"
    and platform.system() == "Darwin"
    and platform.machine() == "x86_64"
    and np.__version__ in _KNOWN_AFFECTED_NUMPY_VERSIONS
    and np.core._multiarray_umath.__cpu_features__.get("AVX2", False)
)


def _warn_if_bad_numpy(usage):
    if not _IS_BAD_NUMPY:
        return
    msg = (
        f"On Intel macOS, NumPy {np.__version__} from PyPI has a bug in the complex-multiplication"
        f" ufunc that severely affects {usage}."
        " See https://qisk.it/cmul-avx2-numpy-bug for work-around information."
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


from .squ import SingleQubitUnitary
from .uc_pauli_rot import UCPauliRotGate
from .ucrz import UCRZGate
from .ucry import UCRYGate
from .ucrx import UCRXGate
from .diagonal import DiagonalGate
from .uc import UCGate
from .isometry import Isometry
from .initializer import Initialize
