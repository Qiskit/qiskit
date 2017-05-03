"""
Exception for errors raised by unroller.

Author: Andrew Cross
"""


class UnrollerException(Exception):
    """Base class for errors raised by unroller."""

    def __init__(self, *message):
        """Set the error message."""
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
