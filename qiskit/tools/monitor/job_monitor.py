# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""A module for monitoring various qiskit functionality"""

import sys
import time
import threading
from qiskit.qiskiterror import QiskitError

_NOTEBOOK_ENV = False
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    _NOTEBOOK_ENV = True
    from IPython.display import display    # pylint: disable=import-error
    try:
        import ipywidgets as widgets           # pylint: disable=import-error
    except ImportError:
        raise ImportError('These functions  need ipywidgets. '
                          'Run "pip install ipywidgets" before.')
    from qiskit.tools.jupyter.jupyter_magics import _html_checker    # pylint: disable=C0412


def _text_checker(job, interval):
    """A text-based job status checker

    Args:
        job (BaseJob): The job to check.
        interval (int): The interval at which to check.
    """
    status = job.status()
    msg = status.value
    prev_msg = msg
    msg_len = len(msg)

    print('\r%s: %s' % ('Job Status', msg), end='')
    while status.name not in ['DONE', 'CANCELLED', 'ERROR']:
        time.sleep(interval)
        status = job.status()
        msg = status.value

        if status.name == 'QUEUED':
            msg += ' (%s)' % job.queue_position()

        # Adjust length of message so there are no artifacts
        if len(msg) < msg_len:
            msg += ' ' * (msg_len - len(msg))
        elif len(msg) > msg_len:
            msg_len = len(msg)

        if msg != prev_msg:
            print('\r%s: %s' % ('Job Status', msg), end='')
            prev_msg = msg
    print('')


def job_monitor(job, interval=2, monitor_async=False):
    """Monitor the status of a IBMQJob instance.

    Args:
        job (BaseJob): Job to monitor.
        interval (int): Time interval between status queries.
        monitor_async (bool): Monitor asyncronously (in Jupyter only).

    Raises:
        QiskitError: When trying to run async outside of Jupyter
    """
    if _NOTEBOOK_ENV:
        style = "font-size:16px;"
        header = "<p style='{style}'>Job Status: %s </p>".format(style=style)
        status = widgets.HTML(value=header % job.status().value)
        display(status)
        if monitor_async:
            thread = threading.Thread(target=_html_checker, args=(job, interval,
                                                                  status, header))
            thread.start()
        else:
            _html_checker(job, interval, status, header)

    else:
        if monitor_async:
            raise QiskitError('monitor_async only available in Jupyter notebooks.')
        _text_checker(job, interval)
