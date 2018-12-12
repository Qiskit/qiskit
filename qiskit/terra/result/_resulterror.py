# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors when there's an error in the Result
"""

from qiskit import QiskitError


class ResultError(QiskitError):
    """Exceptions raised due to errors in result output.

    It may be better for the Qiskit API to raise this exception.

    Args:
        error (dict): This is the error record as it comes back from
            the API. The format is like::

                error = {'status': 403,
                         'message': 'Your credits are not enough.',
                         'code': 'MAX_CREDITS_EXCEEDED'}
    """
    def __init__(self, error):
        super().__init__(error['message'])
        self.status = error['status']
        self.code = error['code']

    def __str__(self):
        return '{}: {}'.format(self.code, self.message)
