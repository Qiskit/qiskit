# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Initialize the Jupyter routines.
"""

from IPython import get_ipython          # pylint: disable=import-error
from .jupyter_magics import (ProgressBarMagic, StatusMagic)
from .progressbar import HTMLProgressBar, TextProgressBar

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(ProgressBarMagic)
    _IP.register_magics(StatusMagic)
