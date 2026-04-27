# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""List the (qpy_version, qiskit_version, python_versions) triplets for forward-compatibility QPY testing.

For each QPY format version, output all the qiskit versions supporting loading from that QPY version
"""

import packaging.version
from qpy_version_utils import fetch_installable_qiskit_versions

# Minimum Qiskit minor series that can read each supported QPY format version.
QPY_MIN_COMPAT_VERSION = {
    13: "1.4",
    14: "2.0",
    15: "2.1",
    16: "2.2",
    17: "2.3",
}


def get_qiskit_versions():
    """For each minor series referenced in QPY_MIN_COMPAT_VERSION, find the latest
    available patch release on PyPI that is installable on this platform."""
    minimum_minor_by_qpy = {
        qpy_version: packaging.version.parse(minor_key)
        for qpy_version, minor_key in QPY_MIN_COMPAT_VERSION.items()
    }
    needed_minors = set(QPY_MIN_COMPAT_VERSION.values())

    # Fetch all qiskit versions from the minimum supported version onwards
    min_qpy_version = min(minimum_minor_by_qpy.values())
    min_version_str = f"{min_qpy_version.major}.{min_qpy_version.minor}"

    results = {qpy_version: [] for qpy_version in QPY_MIN_COMPAT_VERSION.keys()}

    for version, python_version in fetch_installable_qiskit_versions(
        package="qiskit",
        min_version=min_version_str,
        max_version=None,
    ):
        minor_version = packaging.version.parse(f"{version.major}.{version.minor}")
        minor_key = f"{version.major}.{version.minor}"

        # Skip versions that are not in needed_minors unless they're >= minimum
        if minor_key not in needed_minors and minor_version < min(minimum_minor_by_qpy.values()):
            continue

        # Add this version to all QPY versions it can read
        for qpy_version, minimum_minor in minimum_minor_by_qpy.items():
            if minor_version >= minimum_minor:
                results[qpy_version].append(version)

    return results


def main():
    results = get_qiskit_versions()
    for qpy_version, supported_versions in results.items():
        for supported_version in supported_versions:
            print(f"{qpy_version} {supported_version}")  # noqa: T201


if __name__ == "__main__":
    main()
