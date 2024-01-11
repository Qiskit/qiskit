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
import datetime
import sys
from qiskit.tools.events.pubsub import Subscriber


class BaseProgressBar(Subscriber):
    """An abstract progress bar with some shared functionality."""

    def __init__(self):
        super().__init__()
        self.type = "progressbar"
        self.touched = False
        self.iter = None
        self.t_start = None
        self.t_done = None

    def start(self, iterations):
        """Start the progress bar.

        Parameters:
            iterations (int): Number of iterations.
        """
        self.touched = True
        self.iter = int(iterations)
        self.t_start = time.time()

    def update(self, n):
        """Update status of progress bar."""
        pass

    def time_elapsed(self):
        """Return the time elapsed since start.

        Returns:
            elapsed_time: Time since progress bar started.
        """
        return "%6.2fs" % (time.time() - self.t_start)

    def time_remaining_est(self, completed_iter):
        """Estimate the remaining time left.

        Parameters:
            completed_iter (int): Number of iterations completed.

        Returns:
            est_time: Estimated time remaining.
        """
        if completed_iter:
            t_r_est = (time.time() - self.t_start) / completed_iter * (self.iter - completed_iter)
        else:
            t_r_est = 0
        date_time = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=t_r_est)
        time_string = "%02d:%02d:%02d:%02d" % (
            date_time.day - 1,
            date_time.hour,
            date_time.minute,
            date_time.second,
        )

        return time_string

    def finished(self):
        """Run when progress bar has completed."""
        pass


class TextProgressBar(BaseProgressBar):
    """
    A simple text-based progress bar.

    output_handler : the handler the progress bar should be written to, default
                     is sys.stdout, another option is sys.stderr
    """

    def __init__(self, output_handler=None):
        super().__init__()
        self._init_subscriber()

        self.output_handler = output_handler if output_handler else sys.stdout

    def _init_subscriber(self):
        def _initialize_progress_bar(num_tasks):
            """ """
            self.start(num_tasks)

        self.subscribe("terra.parallel.start", _initialize_progress_bar)

        def _update_progress_bar(progress):
            """ """
            self.update(progress)

        self.subscribe("terra.parallel.done", _update_progress_bar)

        def _finish_progress_bar():
            """ """
            self.unsubscribe("terra.parallel.start", _initialize_progress_bar)
            self.unsubscribe("terra.parallel.done", _update_progress_bar)
            self.unsubscribe("terra.parallel.finish", _finish_progress_bar)
            self.finished()

        self.subscribe("terra.parallel.finish", _finish_progress_bar)

    def start(self, iterations):
        self.touched = True
        self.iter = int(iterations)
        self.t_start = time.time()
        pbar = "-" * 50
        self.output_handler.write("\r|{}| {}{}{} [{}]".format(pbar, 0, "/", self.iter, ""))

    def update(self, n):
        # Don't update if we are not initialized or
        # the update iteration number is greater than the total iterations set on start.
        if not self.touched or n > self.iter:
            return
        filled_length = int(round(50 * n / self.iter))
        pbar = "█" * filled_length + "-" * (50 - filled_length)
        time_left = self.time_remaining_est(n)
        self.output_handler.write("\r|{}| {}{}{} [{}]".format(pbar, n, "/", self.iter, time_left))
        if n == self.iter:
            self.output_handler.write("\n")
        self.output_handler.flush()
