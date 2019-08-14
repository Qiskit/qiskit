# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module of widgets for job monitoring"""
import sys
import time
import threading


def _job_monitor(job, status, watcher):
    """Monitor the status of a IBMQJob instance.

    Args:
        job (BaseJob): Job to monitor.
        status (Enum): Job status.
        watcher (JobWatcher): Job watcher instance
    """
    thread = threading.Thread(target=_job_checker, args=(job, status, watcher))
    thread.start()


def _job_checker(job, status, watcher):
    """A simple job status checker

    Args:
        job (BaseJob): The job to check.
        status (Enum): Job status.
        watcher (JobWatcher): Job watcher instance

    """
    prev_status_name = None
    prev_queue_pos = None
    interval = 2
    exception_count = 0

    while status.name not in ['DONE', 'CANCELLED', 'ERROR']:
        time.sleep(interval)
        try:
            status = job.status()
            exception_count = 0

            if status.name == 'QUEUED':
                queue_pos = job.queue_position()
                if queue_pos != prev_queue_pos:

                    update_info = (job.job_id(), status.name,
                                   queue_pos, status.value)

                    watcher.update_single_job(update_info)
                    interval = max(queue_pos, 2)
                    prev_queue_pos = queue_pos

            elif status.name != prev_status_name:
                update_info = (job.job_id(), status.name, 0, status.value)

                watcher.update_single_job(update_info)
                interval = 2
                prev_status_name = status.name

        # pylint: disable=broad-except
        except Exception:
            exception_count += 1
            if exception_count == 5:
                update_info = (job.job_id(), 'NA', 0, "Could not query job.")
                watcher.update_single_job(update_info)
                sys.exit()
