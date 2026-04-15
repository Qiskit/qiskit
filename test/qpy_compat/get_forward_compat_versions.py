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

For each QPY format version, output all the qiskit versions supporting loading from that QPY version"""

import json
import re
import sys
import urllib.request

import packaging.version
import packaging.tags

import qiskit

# Which QPY format version can be *read* by which Qiskit minor series.
QPY_COMPAT_MATRIX = {
    13: ["1.4", "2.0", "2.1", "2.2", "2.3", "2.4"],
    14: ["2.0", "2.1", "2.2", "2.3", "2.4"],
    15: ["2.1", "2.2", "2.3", "2.4"],
    16: ["2.2", "2.3", "2.4"],
    17: ["2.3", "2.4"],
}


def tags_from_wheel_name(wheel: str):
    """Extract the wheel tag from its filename."""
    assert wheel.lower().endswith(".whl")
    # For more information, see:
    # - https://packaging.python.org/en/latest/specifications/binary-distribution-format/
    # - https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
    #
    # In particular, take note that a wheel's filename can include "compressed tag sets" (and our
    # Linux wheels generally do this), which is why this function returns an iterable of tags.
    _prefix, interpreters, abis, platforms = wheel[:-4].rsplit("-", 3)
    yield from (
        packaging.tags.Tag(interpreter, abi, platform)
        for interpreter in interpreters.split(".")
        for abi in abis.split(".")
        for platform in platforms.split(".")
    )


def get_qiskit_versions():
    """For each minor series referenced in QPY_COMPAT_MATRIX, find the latest
    available patch release on PyPI that is installable on this platform."""
    our_version = packaging.version.parse(qiskit.__version__)
    supported_tags = set(packaging.tags.sys_tags())
    no_docker = "--no-docker" in sys.argv

    needed_minors = set()
    for minors in QPY_COMPAT_MATRIX.values():
        needed_minors.update(minors)

    with urllib.request.urlopen("https://pypi.org/pypi/qiskit/json") as fd:
        data = json.load(fd)

    # best[minor_key] = (parsed_version, python_version_str)
    results = {qpy_version: [] for qpy_version in QPY_COMPAT_MATRIX.keys()} 
    for raw_version, payload in data["releases"].items():
        version = packaging.version.parse(raw_version)
        if version > our_version:
            continue
        if version.pre is not None and version.pre[0] in ("a", "b"):
            continue
        minor_key = f"{version.major}.{version.minor}"
        if minor_key not in needed_minors:
            continue

        if no_docker:
            if not any(
                tag in supported_tags
                for release in payload
                if release["packagetype"] == "bdist_wheel" and not release["yanked"]
                for tag in tags_from_wheel_name(release["filename"])
            ):
                continue
            python_version = ""
        else:
            try:
                python_versions = [
                    release["python_version"]
                    for release in payload
                    if release["packagetype"] == "bdist_wheel" and not release["yanked"]
                ]
                python_versions = [
                    re.sub(r"^cp(\d)(\d+)$", r"\1.\2", v) for v in python_versions
                ]
                python_version = max(
                    python_versions, key=lambda s: tuple(map(int, s.split(".")))
                )
            except ValueError:
                print(
                    f"skipping '{version}', no installable binary artifacts",
                    file=sys.stderr,
                )
                continue
        for qpy_version, supported_minor_keys in QPY_COMPAT_MATRIX.items():
            if minor_key in supported_minor_keys:
                results[qpy_version].append((version, python_version))
    return results


def main():
    results = get_qiskit_versions()
    for qpy_version, supported_versions in results.items():
        for (supported_version, python_version) in supported_versions:
            print(
                f"{qpy_version} {supported_version} {python_version}"
            )

if __name__ == "__main__":
    main()

