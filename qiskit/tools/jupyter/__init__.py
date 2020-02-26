# -*- coding: utf-8 -*-

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
===========================================
Jupyter Tools (:mod:`qiskit.tools.jupyter`)
===========================================

.. currentmodule:: qiskit.tools.jupyter

A Collection of Jupyter magic functions and tools
that extend the functionality of Qiskit.

HTMLProgressBar
===============

.. jupyter-execute::

    import numpy as np
    from qiskit.tools.parallel import parallel_map
    import qiskit.tools.jupyter

    %qiskit_progress_bar
    parallel_map(np.sin, np.linspace(0,10,100));


Qiskit version table
====================

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_version_table


Qiskit copyright
================

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright

"""
import warnings

from IPython import get_ipython          # pylint: disable=import-error
from .jupyter_magics import ProgressBarMagic
from .progressbar import HTMLProgressBar
from .version_table import VersionTable
from .copyright import Copyright

try:
    import qiskit.providers.ibmq.jupyter
except ImportError:
    pass

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(ProgressBarMagic)
    _IP.register_magics(VersionTable)
    _IP.register_magics(Copyright)
