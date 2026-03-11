# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
================================================
C API interface from Python (:mod:`qiskit.capi`)
================================================

.. currentmodule:: qiskit.capi

This module provides Python-space interactions with Qiskit's public C API.

For documentation on the C API itself, see `Qiskit C API
<https://quantum.cloud.ibm.com/docs/api/qiskit-c/>`__.

Build-system interaction
========================

The Python package :mod:`qiskit` contains all of the Qiskit C API header files, and a compiled
shared-object library that includes all of the C-API functions.  You can access the locations of
these two objects with the functions :func:`get_include` and :func:`get_lib` respectively.

.. warning::
    You typically *should not* link directly against the output of :func:`get_lib`, unless you know
    what you are doing.  In particular, directly linking against this objcet is not a safe way to
    build a distributable Python extension module that uses Qiskit's C API.

    However, if you understand all the caveats of direct linking, you can use the function to get
    the location of the library.

.. autofunction:: get_include
.. autofunction:: get_lib
"""

__all__ = ["get_include", "get_lib"]

from pathlib import Path

import qiskit._accelerate


def get_include() -> str:
    """Get the directory containing the ``qiskit.h`` C header file and the internal
    ``qiskit/*.h`` auxiliary files.

    When using Qiskit as a build dependency, you typically want to include this directory on the
    include search path of your compiler, such as:

    .. code-block:: bash

        qiskit_include=$(python -c 'import qiskit.capi; print(qiskit.capi.get_include())')
        gcc -I "$qiskit_include" my_bin.c -o my_bin

    The location of this directory within the Qiskit package data is not fixed, and may change
    between Qiskit versions.  You should always use this function to retrieve the directory.

    Returns:
        an absolute path to the package include-files directory.
    """
    return str(Path(__file__).parent.absolute() / "include")


def get_lib() -> str:
    """Get the path to a shared-object library that contains all the C-API exported symbols.

    .. warning::
        You typically *should not* link directly against this object.  In particular, directly
        linking against this object is not a safe way to build a Python extension module that uses
        Qiskit's C API.

    You can, if you choose, use :mod:`ctypes` to access the C API symbols contained in this object,
    though beware that the C-API types declared in the header file are not interchangeable with the
    Python objects that correspond to them.

    The location and name of this file within the Qiskit package data is not fixed, and may change
    between Qiskit versions.

    Returns:
        an absolute path to the shared-object library containing the C-API exported symbols.
    """
    return str(Path(qiskit._accelerate.__file__).absolute())
