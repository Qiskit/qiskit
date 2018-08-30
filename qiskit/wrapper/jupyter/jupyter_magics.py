# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A module of magic functions"""

import sys
import time
import threading
from IPython.display import display                              # pylint: disable=import-error
from IPython.core import magic_arguments                         # pylint: disable=import-error
from IPython.core.magic import cell_magic, Magics, magics_class  # pylint: disable=import-error
import ipywidgets as widgets                                     # pylint: disable=import-error
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
    def qiskit_job_status(self, line='', cell=None):
        """A Jupyter magic function to check the status of a Qiskit job instance.
        """
        args = magic_arguments.parse_argstring(self.qiskit_job_status, line)
        # Split cell lines to get LHS variables
        cell_lines = cell.split('\n')
        split_lines = []
        for cline in cell_lines:
            split_lines.append(cline.replace(' ', '').split('='))

        # Execute the cell
        self.shell.ex(cell)

        # Look for all vars that are BaseJob instances
        jobs = []
        for spline in split_lines:
            if len(spline) == 2:
                line_var = spline[0]
                # Check corner case where expression with equals is commented out
                if '#' not in line_var:
                    if isinstance(self.shell.user_ns[line_var], qiskit.backends.basejob.BaseJob):
                        jobs.append(line_var)
        # Must have one job class
        if not any(jobs):
            raise Exception(
                "Cell just contain at least one 'job=qiskit.execute(...)' expression")

        def _checker(job_var, status, header):
            job_status_name = job_var.status().name
            while job_status_name not in ['DONE', 'CANCELLED']:
                time.sleep(args.interval)
                job_status = job_var.status()
                job_status_name = job_status.name
                job_status_msg = job_status.value
                if job_status_name == 'ERROR':
                    break
                else:
                    if job_status_name == 'QUEUED':
                        job_status_msg += ' (%s)' % job_var._queue_position
                    status.value = header % (job_status_msg)

            status.value = header % (job_status_msg)
            # Explicitly stop the thread just to be safe.
            sys.exit()

        # List index of job if checking status of multiple jobs.
        multi_job = False
        if len(jobs) > 1:
            multi_job = True

        job_checkers = []
        # Loop over every BaseJob that was found.
        for idx, job in enumerate(jobs):
            job_var = self.shell.user_ns[job]
            style = "font-size:16px;"
            if multi_job:
                idx_str = '[%s]' % idx
            else:
                idx_str = ''
            header = "<p style='{style}'>Job Status {id}: %s </p>".format(id=idx_str,
                                                                          style=style)
            status = widgets.HTML(
                value=header % job_var.status().value)

            thread = threading.Thread(target=_checker, args=(job_var, status, header))
            thread.start()
            job_checkers.append(status)

        # Group all HTML widgets into single vertical layout
        box = widgets.VBox(job_checkers)
        display(box)
