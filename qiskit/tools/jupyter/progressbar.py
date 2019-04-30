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

"""Progress bars module"""

import time
try:
    import ipywidgets as widgets           # pylint: disable=import-error
except ImportError:
    raise ImportError('These functions  need ipywidgets. '
                      'Run "pip install ipywidgets" before.')
from IPython.display import display         # pylint: disable=import-error
from qiskit.tools.events.progressbar import BaseProgressBar


class HTMLProgressBar(BaseProgressBar):
    """
    A simple HTML progress bar for using in IPython notebooks.
    """
    def __init__(self):
        super().__init__()
        self.progress_bar = None
        self.label = None
        self.box = None
        self._init_subscriber()

    def _init_subscriber(self):
        def _initialize_progress_bar(num_tasks):
            """ When an event of compilation starts, this function will be called, and
            will initialize the progress bar.

            Args
                num_tasks: Number of compilation tasks the progress bar will track
            """
            self.start(num_tasks)
        self.subscribe("terra.parallel.start", _initialize_progress_bar)

        def _update_progress_bar(progress):
            """ When an event of compilation completes, this function will be called, and
            will update the progress bar indication.

            Args
                progress: Number of tasks completed
            """
            self.update(progress)
        self.subscribe("terra.parallel.done", _update_progress_bar)

        def _finish_progress_bar():
            """ When an event of compilation finishes (meaning that there's no more circuits to
            compile), this function will be called, unsubscribing from all events and
            finishing the progress bar."""
            self.unsubscribe("terra.parallel.start", _initialize_progress_bar)
            self.unsubscribe("terra.parallel.done", _update_progress_bar)
            self.unsubscribe("terra.parallel.finish", _finish_progress_bar)
            self.finished()
        self.subscribe("terra.parallel.finish", _finish_progress_bar)

    def start(self, iterations):
        self.touched = True
        self.iter = int(iterations)
        self.t_start = time.time()
        self.progress_bar = widgets.IntProgress(min=0, max=self.iter, value=0)
        self.progress_bar.bar_style = 'info'
        self.label = widgets.HTML()
        self.box = widgets.VBox(children=[self.label, self.progress_bar])
        display(self.box)

    def update(self, n):
        self.progress_bar.value += 1
        lbl = "Completed %s/%s: Est. remaining time: %s."
        self.label.value = lbl % (n, self.iter, self.time_remaining_est(n))

    def finished(self):
        self.t_done = time.time()
        self.progress_bar.bar_style = 'success'
        self.label.value = "Elapsed time: %s" % self.time_elapsed()
