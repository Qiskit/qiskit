"""
Exception for errors raised by unroller backends.

Author: Andrew Cross
"""


class BackendException(Exception):
    """Base class for errors raised by unroller backends."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
