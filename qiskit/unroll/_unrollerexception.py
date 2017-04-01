"""
Exception for errors raised by unroller.

Author: Andrew Cross
"""


class UnrollerException(Exception):
    """Base class for errors raised by unroller."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
