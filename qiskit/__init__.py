# -*- coding: utf-8 -*-

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

# pylint: disable=wrong-import-order,invalid-name,wrong-import-position


"""Main Qiskit public functionality."""

import pkgutil
import sys
import warnings
import os

# First, check for required Python and API version
from . import util

# qiskit errors operator
from qiskit.exceptions import QiskitError

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
__path__ = pkgutil.extend_path(__path__, __name__)

# Please note these are global instances, not modules.
from qiskit.providers.basicaer import BasicAer

_config = _user_config.get_config()

# Try to import the Aer provider if installed.
try:
    from qiskit.providers.aer import Aer
except ImportError:
    suppress_warnings = os.environ.get('QISKIT_SUPPRESS_PACKAGING_WARNINGS', '')
    if suppress_warnings.upper() != 'Y':
        if not _config.get('suppress_packaging_warnings') or suppress_warnings.upper() == 'N':
            warnings.warn('Could not import the Aer provider from the qiskit-aer '
                          'package. Install qiskit-aer or check your installation.',
                          RuntimeWarning)
# Try to import the IBMQ provider if installed.
try:
    from qiskit.providers.ibmq import IBMQ
except ImportError:
    suppress_warnings = os.environ.get('QISKIT_SUPPRESS_PACKAGING_WARNINGS', '')
    if suppress_warnings.upper() != 'Y':
        if not _config.get('suppress_packaging_warnings') or suppress_warnings.upper() == 'N':
            warnings.warn('Could not import the IBMQ provider from the '
                          'qiskit-ibmq-provider package. Install '
                          'qiskit-ibmq-provider or check your installation.',
                          RuntimeWarning)

# Moved to after IBMQ and Aer imports due to import issues
# with other modules that check for IBMQ (tools)
from qiskit.execute import execute  # noqa
from qiskit.compiler import transpile, assemble, schedule  # noqa

from .version import __version__  # noqa
from .version import _get_qiskit_versions  # noqa


if sys.version_info[0] == 3 and sys.version_info[1] == 5:
    warnings.warn(
        "Using Qiskit with Python 3.5 is deprecated as of the 0.12.0 release. "
        "Support for running Qiskit with Python 3.5 will be removed at the "
        "Python 3.5 EoL on 09/13/2020.", DeprecationWarning)


__qiskit_version__ = _get_qiskit_versions()
