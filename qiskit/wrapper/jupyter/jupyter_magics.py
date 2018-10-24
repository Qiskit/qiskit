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
from qiskit.transpiler._receiver import receiver as rec
from qiskit.transpiler._progressbar import TextProgressBar
from qiskit.wrapper.jupyter.progressbar import HTMLProgressBar


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
        line_vars = []
        for cline in cell_lines:
            if '=' in cline and '==' not in cline:
                line_vars.append(cline.replace(' ', '').split('=')[0])
            elif '.append(' in cline:
                line_vars.append(cline.replace(' ', '').split('(')[0])

        # Execute the cell
        self.shell.ex(cell)

        # Look for all vars that are BaseJob instances
        jobs = []
        for var in line_vars:
            iter_var = False
            if '#' not in var:
                # The line var is a list or array, but we cannot parse the index
                # so just iterate over the whole array for jobs.
                if '[' in var:
                    var = var.split('[')[0]
                    iter_var = True
                elif '.append' in var:
                    var = var.split('.append')[0]
                    iter_var = True

                if iter_var:
                    for item in self.shell.user_ns[var]:
                        if isinstance(item, qiskit.backends.basejob.BaseJob):
                            jobs.append(item)
                else:
                    if isinstance(self.shell.user_ns[var],
                                  qiskit.backends.basejob.BaseJob):
                        jobs.append(self.shell.user_ns[var])

        # Must have one job class
        if not any(jobs):
            raise Exception(
                "Cell must contain at least one variable of BaseJob type.")

        def _checker(job_var, status, header):
            job_status = job_var.status()
            job_status_name = job_status.name
            job_status_msg = job_status.value
            status.value = header % (job_status_msg)
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
        for idx, job_var in enumerate(jobs):
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


@magics_class
class ProgressBarMagic(Magics):
    """A class of progress bar magic functions.
    """
    @cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-t',
        '--type',
        type=str,
        default='html',
        help="Type of progress bar, 'html' or 'text'."
    )
    def qiskit_progress_bar(self, line='', cell=None):  # pylint: disable=W0613
        """A Jupyter magic function to generate progressbar.
        """
        args = magic_arguments.parse_argstring(self.qiskit_progress_bar, line)
        if args.type == 'html':
            progress_bar = HTMLProgressBar()
        elif args.type == 'text':
            progress_bar = TextProgressBar()
        else:
            raise qiskit.QISKitError('Invalid progress bar type.')
        self.shell.ex(cell)
        # Remove progress bar from receiver if not used in cell
        if progress_bar.channel_id in rec.channels.keys():
            rec.remove_channel(progress_bar.channel_id)
