# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions for classiq drawer"""


class ClassiqQASMException(Exception):
    """Raised when failed to send the qasm code to Classiq

    Args:
        status_code (int): http status code
        details (str): the request error details
    """

    def __init__(self, status_code: int, details: str):
        error_message = (
            "request to send qasm code to Classiq failed with with status code {}\n {}".format(
                status_code, details
            )
        )
        super().__init__(error_message)


class ClassiqCircuitIDNotFoundException(Exception):
    """Raised when failed to retrieve circuit_id from Classiq

    Args:
        code (int): http status code
    """

    def __init__(self, code: int):
        error_message = "Failed to retrieve circuit ID with status code {}".format(code)
        super().__init__(error_message)
