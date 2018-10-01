# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""A module for monitoring various qiskit functionality"""

import sys
import time
import threading


def job_monitor(job, interval=2, monitor_async=False):
    """Monitor the status of a IBMQJob instance.

    Args:
        job (BaseJob): Job to monitor.
        interval (int): Time interval between status queries.
        monitor_async (bool): Monitor asyncronously.

    Notes:
        This function blocks output until the job has completed,
        and is therefore useful in scripts, otherwise use the
        Jupyter notebook magic %%qiskit_job_status.
    """

    def _checker(job, interval):
        status = job.status()
        msg = status.value
        prev_msg = msg
        msg_len = len(msg)

        sys.stdout.write('\r%s: %s' % ('Job Status', msg))
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
                sys.stdout.write('\r%s: %s' % ('Job Status', msg))
                prev_msg = msg
        sys.exit()
    if monitor_async:
        thread = threading.Thread(target=_checker, args=(job, interval))
        thread.start()
    else:
        _checker(job, interval)
