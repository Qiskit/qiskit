"""
Exception for errors raised by the bloch module.

Author: Andrew Cross
"""


class BlochError(Exception):
    """Base class for errors raised by the bloch module."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
