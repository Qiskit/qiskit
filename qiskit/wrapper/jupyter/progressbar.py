# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""An HTML progress bar for Jupyter notebooks"""

import time
import ipywidgets as widgets                # pylint: disable=import-error
from IPython.display import display         # pylint: disable=import-error
import qiskit.transpiler._receiver as rec
from qiskit.transpiler._progressbar import BaseProgressBar


class HTMLProgressBar(BaseProgressBar):
    """
    A simple HTML progress bar for using in IPython notebooks.
    """
    def __init__(self):
        super().__init__()
        self.progress_bar = None
        self.label = None
        self.box = None

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
        rec.receiver.remove_channel(self.channel_id)
