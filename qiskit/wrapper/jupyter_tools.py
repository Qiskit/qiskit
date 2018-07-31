# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""Tools for working in Jupyter notebooks"""

import sys
import uuid
import time
from qiskit.wrapper.progressbar import BaseProgressBar

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    from IPython.display import (HTML, Javascript, display)
    __all__ = ['HTMLProgressBar']
else:
    __all__ = []


class HTMLProgressBar(BaseProgressBar):
    """
    A simple HTML progress bar for using in IPython notebooks.

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = HTMLProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update(n)
            compute_with_n(n)
    """

    def __init__(self):
        super().__init__()
        self.divid = str(uuid.uuid4())
        self.textid = str(uuid.uuid4())
        self.boarder = HTML("""\
<div style="border: 2px solid grey; width: 600px">
  <div id="%s" \
style="background-color: rgba(121,195,106,0.75); width:0%%">&nbsp;</div>
</div>
<p id="%s"></p>
""" % (self.divid, self.textid))
        display(self.boarder)

    def update(self, n):
        percent = (n / self.iter) * 100.0
        if percent >= self.p_chunk:
            lbl = ("Elapsed time: %s. " % self.time_elapsed() +
                   "Est. remaining time: %s." % self.time_remaining_est(percent))
            js_code = ("$('div#%s').width('%i%%');" % (self.divid, percent) +
                       "$('p#%s').text('%s');" % (self.textid, lbl))
            display(Javascript(js_code))
            # display(Javascript("$('div#%s').width('%i%%')" % (self.divid,
            # p)))
            self.p_chunk += self.p_chunk_size

    def finished(self):
        self.t_done = time.time()
        lbl = "Elapsed time: %s" % self.time_elapsed()
        js_code = ("$('div#%s').width('%i%%');" % (self.divid, 100.0) +
                   "$('p#%s').text('%s');" % (self.textid, lbl))
        display(Javascript(js_code))
