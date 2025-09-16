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

"""Compatibility helpers for the Numpy 1.x to 2.0 transition."""

import re
import typing
import warnings

import numpy as np

# This version pattern is taken from the pypa packaging project:
# https://github.com/pypa/packaging/blob/21.3/packaging/version.py#L223-L254 which is dual licensed
# Apache 2.0 and BSD see the source for the original authors and other details.
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION = np.lib.NumpyVersion(np.__version__)
VERSION_PARTS: typing.Tuple[int, ...]
"""The numeric parts of the Numpy release version, e.g. ``(2, 0, 0)``.  Does not include pre- or
post-release markers (e.g. ``rc1``)."""
if match := re.fullmatch(_VERSION_PATTERN, np.__version__, flags=re.VERBOSE | re.IGNORECASE):
    # Assuming Numpy won't ever introduce epochs, and we don't care about pre/post markers.
    VERSION_PARTS = tuple(int(x) for x in match["release"].split("."))
else:
    # Just guess a version.  We know all existing Numpys have good version strings, so the only way
    # this should trigger is from a new or a dev version.
    warnings.warn(
        f"Unrecognized version string for Numpy: '{np.__version__}'.  Assuming Numpy 2.0.",
        RuntimeWarning,
    )
    VERSION_PARTS = (2, 0, 0)

COPY_ONLY_IF_NEEDED = None if VERSION_PARTS >= (2, 0, 0) else False
"""The sentinel value given to ``np.array`` and ``np.ndarray.astype`` (etc) to indicate that a copy
should be made only if required."""
