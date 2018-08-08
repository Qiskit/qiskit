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
        default=1,
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
                if isinstance(self.shell.user_ns[_line_var], qiskit.backends.basejob.BaseJob):
                    _jobs.append(_line_var)
        # Must have one job class
        if not any(_jobs):
            raise Exception(
                "Cell just contain at least one 'job=qiskit.execute(...)' expression")
        # Cannot have more than one job class
        elif len(_jobs) != 1:
            raise Exception("Cell can have only a single job class instance.")

        _job_var = self.shell.user_ns[_jobs[0]]

        _style = "font-size:16px;"
        _header = "<p style='{style}'>Job Status: %s </p>".format(style=_style)
        status = widgets.HTML(
            value=_header % _job_var.status['status_msg'])
        display(status)

        def _checker(status):
            while _job_var.status['status'].name != 'DONE':
                time.sleep(args.interval)
                _status = _job_var.status
                _status_name = _status['status'].name
                _status_msg = _status['status_msg']
                if _status_name == 'ERROR':
                    break
                else:
                    if _status_name == 'QUEUED':
                        _status_msg += ' (%s)' % _status['queue_position']
                    status.value = _header % _status_msg

            status.value = _header % _job_var.status['status_msg']
            # Explicitly stop the thread just to be safe.
            sys.exit()

        thread = threading.Thread(target=_checker, args=(status,))
        thread.start()
