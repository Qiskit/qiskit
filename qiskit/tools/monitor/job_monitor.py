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


"""A module for monitoring various qiskit functionality"""

import sys
import time
import threading
from qiskit.exceptions import QiskitError

_NOTEBOOK_ENV = False
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    _NOTEBOOK_ENV = True
    from IPython.display import display    # pylint: disable=import-error


def _text_checker(job, interval, _interval_set=False, quiet=False, output=sys.stdout):
    """A text-based job status checker

    Args:
        job (BaseJob): The job to check.
        interval (int): The interval at which to check.
        _interval_set (bool): Was interval time set by user?
        quiet (bool): If True, do not print status messages.
        output (file): The file like object to write status messages to.
        By default this is sys.stdout.

    """
    status = job.status()
    msg = status.value
    prev_msg = msg
    msg_len = len(msg)

    if not quiet:
        print('\r%s: %s' % ('Job Status', msg), end='', file=output)
    while status.name not in ['DONE', 'CANCELLED', 'ERROR']:
        time.sleep(interval)
        status = job.status()
        msg = status.value

        if status.name == 'QUEUED':
            msg += ' (%s)' % job.queue_position()
            if not _interval_set:
                interval = max(job.queue_position(), 2)
        else:
            if not _interval_set:
                interval = 2

        # Adjust length of message so there are no artifacts
        if len(msg) < msg_len:
            msg += ' ' * (msg_len - len(msg))
        elif len(msg) > msg_len:
            msg_len = len(msg)

        if msg != prev_msg and not quiet:
            print('\r%s: %s' % ('Job Status', msg), end='', file=output)
            prev_msg = msg
    if not quiet:
        print('', file=output)


def job_monitor(job, interval=None, monitor_async=False, quiet=False, output=sys.stdout):
    """Monitor the status of a IBMQJob instance.

    Args:
        job (BaseJob): Job to monitor.
        interval (int): Time interval between status queries.
        monitor_async (bool): Monitor asyncronously (in Jupyter only).
        quiet (bool): If True, do not print status messages.
        output (file): The file like object to write status messages to.
        By default this is sys.stdout.

    Raises:
        QiskitError: When trying to run async outside of Jupyter
        ImportError: ipywidgets not available for notebook.
    """
    if interval is None:
        _interval_set = False
        interval = 2
    else:
        _interval_set = True
    if _NOTEBOOK_ENV:
        if monitor_async:
            try:
                import ipywidgets as widgets  # pylint: disable=import-error
            except ImportError:
                raise ImportError('These functions  need ipywidgets. '
                                  'Run "pip install ipywidgets" before.')
            from qiskit.tools.jupyter.jupyter_magics import _html_checker  # pylint: disable=C0412

            style = "font-size:16px;"
            header = "<p style='{style}'>Job Status: %s </p>".format(
                style=style)
            status = widgets.HTML(value=header % job.status().value)
            display(status)

            thread = threading.Thread(target=_html_checker, args=(job, interval,
                                                                  status, header))
            thread.start()
        else:
            _text_checker(job, interval, _interval_set,
                          quiet=quiet, output=output)

    else:
        if monitor_async:
            raise QiskitError(
                'monitor_async only available in Jupyter notebooks.')
        _text_checker(job, interval, _interval_set, quiet=quiet, output=output)
