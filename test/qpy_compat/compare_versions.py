#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compare Qiskit versions to determine if we should run qpy compat tests."""

import argparse
import sys

from qiskit.qpy.interface import VERSION_PATTERN_REGEX


def main():
    """Main function."""
    parser = argparse.ArgumentParser(prog="compare_version", description="Compare version strings")
    parser.add_argument(
        "source_version", help="Source version of Qiskit that is generating the payload"
    )
    parser.add_argument("test_version", help="Version under test that will load the QPY payload")
    args = parser.parse_args()
    source_match = VERSION_PATTERN_REGEX.search(args.source_version)
    target_match = VERSION_PATTERN_REGEX.search(args.test_version)
    source_version = tuple(int(x) for x in source_match.group("release").split("."))
    target_version = tuple(int(x) for x in target_match.group("release").split("."))
    if source_version > target_version:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
