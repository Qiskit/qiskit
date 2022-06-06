# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Import numba if available, otherwise, dummy functions.
"""


def _noop_jit(f, *args, **kwargs):
    """A a decorator that does nothing"""
    return f


def _have_numba():
    try:
        import numba

        return True
    except ImportError:
        return False


# True if importing numba succeeded
HAVE_NUMBA = _have_numba()
# HAVE_NUMBA = False

if HAVE_NUMBA:
    from numba import vectorize, jit, njit, int64
    from numba.typed import List
else:
    njit = _noop_jit
    jit = _noop_jit
    vectorize = _noop_jit
    int64 = None
    List = None
