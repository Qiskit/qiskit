# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Exception for errors raised by the JobProcessor when there's an error
in the result
"""

from ._qiskiterror import QISKitError


class ResultError(QISKitError):
    """Exceptions raised due to errors in result output.

    It may be better for the QISKit API to raise this exception.

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
