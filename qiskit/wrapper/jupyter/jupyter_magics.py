# -*- coding: utf-8 -*-
# pylint: disable=import-error
# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A module of magic functions"""

import sys
import time
import threading
from IPython.display import display
from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, Magics, magics_class
import ipywidgets as widgets
import qiskit


@magics_class
class StatusMagic(Magics):
    """A class of status magic functions.
    """
    @cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-i',
        '--interval',
        type=float,
        default=2,
        help='Interval for status check.'
    )
    def qiskit_job_status(self, line='', cell=None):  # pylint: disable=W0613
        """A Jupyter magic function to check the status of a Qiskit job instance.
        """
        args = magic_arguments.parse_argstring(self.qiskit_job_status, line)
        # Split cell lines to get LHS variables
        _cell_lines = cell.split('\n')
        _split_lines = []
        for _line in _cell_lines:
            _split_lines.append(_line.replace(' ', '').split('='))

        # Execute the cell
        self.shell.ex(cell)

        # Look for all vars that are BaseJob instances
        _jobs = []
        for _line in _split_lines:
            if len(_line) == 2:
                _line_var = _line[0]
                # Check corner case where expression with equals is commented out
                if '#' not in _line_var:
                    if isinstance(self.shell.user_ns[_line_var], qiskit.backends.basejob.BaseJob):
                        _jobs.append(_line_var)
        # Must have one job class
        if not any(_jobs):
            raise Exception(
                "Cell just contain at least one 'job=qiskit.execute(...)' expression")

        def _checker(job_var, status, header):
            _status_name = job_var.status().name
            while _status_name not in ['DONE', 'CANCELLED']:
                time.sleep(args.interval)
                _status = job_var.status()
                _status_name = _status.name
                _status_msg = _status.value
                if _status_name == 'ERROR':
                    break
                else:
                    if _status_name == 'QUEUED':
                        _status_msg += ' (%s)' % job_var._queue_position
                    status.value = header % (_status_msg)

            status.value = header % (_status_msg)
            # Explicitly stop the thread just to be safe.
            sys.exit()

        # List index of job if checking status of multiple jobs.
        _multi_job = False
        if len(_jobs) > 1:
            _multi_job = True

        _job_checkers = []
        # Loop over every BaseJob that was found.
        for idx, job in enumerate(_jobs):
            job_var = self.shell.user_ns[job]
            _style = "font-size:16px;"
            if _multi_job:
                idx_str = '[%s]' % idx
            else:
                idx_str = ''
            _header = "<p style='{style}'>Job Status {id}: %s </p>".format(id=idx_str,
                                                                           style=_style)
            _status = widgets.HTML(
                value=_header % job_var.status().value)

            thread = threading.Thread(target=_checker, args=(job_var, _status, _header))
            thread.start()
            _job_checkers.append(_status)

        # Group all HTML widgets into single vertical layout
        box = widgets.VBox(_job_checkers)
        display(box)
