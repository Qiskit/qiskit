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

"""Progress bars module"""

import time
import datetime
import sys
from qiskit.wrapper.receiver import receiver as rec


class BaseProgressBar(object):
    """An abstract progress bar with some shared functionality.
    """

    def __init__(self):
        self.type = 'progressbar'
        self.touched = False
        self.channel_id = rec.add_channel(self)
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
        """Update status of progress bar.
        """
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
        percent = self.iter / completed_iter
        if percent > 0:
            t_r_est = (time.time() - self.t_start) * (percent - 1)
        else:
            t_r_est = 0

        date_time = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=t_r_est)
        time_string = "%02d:%02d:%02d:%02d" % \
            (date_time.day - 1, date_time.hour, date_time.minute, date_time.second)

        return time_string

    def finished(self):
        """Run when progress bar has completed.
        """
        rec.remove_channel(self.channel_id)


class TextProgressBar(BaseProgressBar):
    """
    A simple text-based progress bar.
    """
    def __init__(self):
        super().__init__()
        self.p_chunk = 10
        self.p_chunk_size = 10

    def update(self, n):
        percent = (n / self.iter) * 100.0
        if percent >= self.p_chunk:
            print("%4.1f%%." % percent +
                  " Run time: %s." % self.time_elapsed() +
                  " Est. time left: %s" % self.time_remaining_est(n))
            sys.stdout.flush()
            self.p_chunk += self.p_chunk_size

    def finished(self):
        rec.remove_channel(self.channel_id)
