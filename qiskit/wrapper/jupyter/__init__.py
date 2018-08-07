# -*- coding: utf-8 -*-
# pylint: disable=import-error
# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Initilize the Jupyter routines.
"""

from IPython import get_ipython
from .jupyter_magics import StatusMagic

_ip = get_ipython()  # pylint: disable=C0103
_ip.register_magics(StatusMagic)

