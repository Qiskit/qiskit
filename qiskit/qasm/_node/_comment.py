# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM comment.
"""
import re

from ._node import Node


class Comment(Node):
    """Node for an OPENQASM comment.
    """

    def __init__(self, value, line):
        """Create the version node."""
        Node.__init__(self, "comment", None, None)
        self.line = line
        self.comment_text = re.match(r'//(.*)$', value).group(1)

    def qasm(self, prec=None):
        """Return the corresponding comment string."""
        # pylint: disable=unused-argument
        return "//%s" % (self.comment_text)
