# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
==================================
Qiskit Tools (:mod:`qiskit.tools`)
==================================

.. currentmodule:: qiskit.tools

Monitoring
----------

A helper module to get IBM backend information and submitted job status.

.. autofunction:: job_monitor
.. autofunction:: backend_monitor
.. autofunction:: backend_overview

.. automodule:: qiskit.tools.events

"""
import importlib
import warnings


_DEPRECATED_NAMES = {
    "parallel_map": "qiskit.utils",
    "parallel": "qiskit.utils.parallel",
}


_DEPRECATED_REMOVALS = {"job_monitor", "backend_monitor", "backend_overview", "progressbar"}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        module_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"Accessing '{name}' from '{__name__}' is deprecated since Qiskit 0.46.0 "
            f"and will be removed in Qiskit 1.0.0. Import from '{module_name}' instead ",
            DeprecationWarning,
            2,
        )
        return getattr(importlib.import_module(module_name), name)
    if name in _DEPRECATED_REMOVALS:
        warnings.warn(
            f"'{name}' has been deprecated and will be removed in Qiskit 1.0.0.",
            DeprecationWarning,
            2,
        )
        return getattr(importlib.import_module(".monitor", "qiskit.tools"), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
