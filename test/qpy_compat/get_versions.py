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

"""List the versions of Qiskit and Terra that should be tested for QPY compatibility."""

import json
import sys
import re
import urllib.request

import packaging.version
import packaging.tags

import qiskit


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


def available_versions():
    """Get all the versions of Qiskit that support exporting QPY, and are installable with the
    active version of Python on this platform."""
    our_version = packaging.version.parse(qiskit.__version__)
    supported_tags = set(packaging.tags.sys_tags())

    def available_versions_for_package(package, min_version=None, max_version=None):
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json") as fd:
            data = json.load(fd)
        min_version = min_version and packaging.version.parse(min_version)
        max_version = max_version and packaging.version.parse(max_version)
        for other_version, payload in data["releases"].items():
            other_version = packaging.version.parse(other_version)
            if min_version is not None and other_version < min_version:
                continue
            if max_version is not None and other_version >= max_version:
                continue
            if other_version > our_version:
                continue
            if other_version.pre is not None and other_version.pre[0] in ("a", "b"):
                # We skip alpha and beta prereleases, but we currently want to test for
                # compatibility with release candidates.
                continue
            # Note: For non-docker runs, this ignores versions that are uninstallable because we're using
            # a Python version that's too new, which can be a problem for the oldest Terras, especially from
            # before we built for abi3.  We're not counting sdists, since if we didn't release a
            # compatible wheel for the current Python version, there's no guarantee it'll install.
            if "--no-docker" in sys.argv:
                if not any(
                    tag in supported_tags
                    for release in payload
                    if release["packagetype"] == "bdist_wheel" and not release["yanked"]
                    for tag in tags_from_wheel_name(release["filename"])
                ):
                    print(
                        f"skipping '{other_version}', which has no installable binary artifacts",
                        file=sys.stderr,
                    )
                    continue
                # we run in no-docker mode, so the python version should be empty
                python_version = ""
            else:  # we run in docker mode, so we need to decide which python version to pull
                try:
                    python_versions = [
                        release["python_version"]
                        for release in payload
                        if release["packagetype"] == "bdist_wheel" and not release["yanked"]
                    ]
                    python_versions = [
                        re.sub(r"^cp(\d)(\d+)$", r"\1.\2", version) for version in python_versions
                    ]  # convert "cp311" to "3.11"
                    python_version = max(
                        python_versions, key=lambda s: tuple(map(int, s.split(".")))
                    )
                except ValueError:
                    print(
                        f"skipping '{other_version}', which has no installable binary artifacts",
                        file=sys.stderr,
                    )
                    continue
            yield (other_version, python_version)

    yield from (
        ("qiskit-terra", version, python_version)
        for version, python_version in available_versions_for_package(
            "qiskit-terra", "0.18.0", "1.0.0"
        )
    )
    yield from (
        ("qiskit", version, python_version)
        for version, python_version in available_versions_for_package("qiskit", "1.0.0")
    )


def main():
    """main"""
    for package, version, python_version in available_versions():
        print(package, version, python_version)


if __name__ == "__main__":
    main()
