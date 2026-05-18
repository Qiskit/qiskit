# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"The Qiskit setup file."

import os
import warnings
from setuptools import setup
from setuptools_rust import Binding, RustExtension

# ==================================================================================================
# Warning: this section is a horrendous monkey-patching hack that re-implements a version of
# https://github.com/PyO3/setuptools-rust/pull/574 on top of `setuptools-rust==1.12.0`, which
# contains just enough functionality for our own use.

from pathlib import Path
import json
import shutil
import functools
import setuptools_rust.build

# This function is only called once in our build, to find the `cdylib` artifact.  It's passed the
# `cargo` messages, which is what we need to locate the `OUT_DIR`; we use this as a hook point to
# intercept them and leak them out for later use.
find_cargo_artifacts_orig = setuptools_rust.build._find_cargo_artifacts
install_extension_orig = setuptools_rust.build.build_rust.install_extension
out_dir = None
generated_files = {"include": "qiskit.capi", "_ctypes.py": "qiskit.capi"}


@functools.wraps(find_cargo_artifacts_orig)
def find_cargo_artifacts_patch(cargo_messages, *, package_id, kinds):
    global out_dir  # noqa: PLW0603 (global) - we have to leak to get the monkeypatch to work.

    # Chances are that the line we're looking for will be the third laste line in the
    # messages.  The last is the completion report, the penultimate is generally the
    # build of the final artifact.
    for message in reversed(cargo_messages):
        if "build-script-executed" not in message or package_id not in message:
            continue
        parsed = json.loads(message)
        if parsed.get("package_id") == package_id:
            out_dir = parsed.get("out_dir")
            break

    return find_cargo_artifacts_orig(cargo_messages, package_id=package_id, kinds=kinds)


@functools.wraps(install_extension_orig)
def install_extension_patch(self, ext, dylib_paths):
    install_extension_orig(self, ext, dylib_paths)

    # `out_dir` and `generated_files` are captured from the outer scope.
    if out_dir is None:
        raise RuntimeError("setup sanity check failed: the cargo `out_dir` is not set")
    build_artifact_dirs = [Path(out_dir)]

    # We'll delegate the finding of the package directories to Setuptools, so we
    # can be sure we're handling editable installs and other complex situations
    # correctly.
    build_ext = self.get_finalized_command("build_ext")
    build_py = self.get_finalized_command("build_py")

    def get_package_dir(package: str) -> Path:
        if self.inplace:
            # If `inplace`, we have to ask `build_py` (like `build_ext` would).
            return Path(build_py.get_package_dir(package))
        # ... If not, `build_ext` knows where to put the package.
        return Path(build_ext.build_lib) / Path(*package.split("."))

    for source, package in generated_files.items():
        dest = get_package_dir(package)
        dest.mkdir(mode=0o755, parents=True, exist_ok=True)
        for artifact_dir in build_artifact_dirs:
            source_full = artifact_dir / source
            dest_full = dest / source_full.name
            if source_full.is_file():
                shutil.copy2(source_full, dest_full)
            elif source_full.is_dir():
                shutil.copytree(source_full, dest_full, dirs_exist_ok=True)
            # This tacitly makes "no match" a silent non-error.


setuptools_rust.build._find_cargo_artifacts = find_cargo_artifacts_patch
setuptools_rust.build.build_rust.install_extension = install_extension_patch

# Normal service resumes below here.
# ==================================================================================================


# Most of this configuration is managed by `pyproject.toml`.  This only includes the extra bits to
# configure `setuptools-rust`, because we do a little dynamic trick with the debug setting, and we
# also want an explicit `setup.py` file to exist so we can manually call
#
#   python setup.py build_rust --inplace --release
#
# to make optimized Rust components even for editable releases, which would otherwise be quite
# unergonomic to do otherwise.


# Check for a default build profile from the environment (`--release` or `--debug` flags to
# `build_rust` override this default).  If not present, we also check if `RUST_DEBUG=1` for
# convenience, since we did that (undocumented) until Qiskit 2.4.
if (build_profile := os.getenv("QISKIT_BUILD_PROFILE", None)) is None:
    rust_debug = os.getenv("RUST_DEBUG", None) == "1" or None
else:
    match build_profile.lower():
        case "debug":
            rust_debug = True
        case "release":
            rust_debug = False
        case _:
            warnings.warn(
                f"QISKIT_BUILD_PROFILE set to unknown value '{build_profile}'."
                " Valid values are 'debug' and 'release'."
            )
            rust_debug = None

# If QISKIT_NO_CACHE_GATES is set then don't enable any features while building
#
# TODO: before final release we should reverse this by default once the default transpiler pass
# is all in rust (default to no caching and make caching an opt-in feature). This is opt-out
# right now to avoid the runtime overhead until we are leveraging the rust gates infrastructure.
if os.getenv("QISKIT_NO_CACHE_GATES") == "1":
    features = []
else:
    features = ["cache_pygates"]

if os.getenv("QISKIT_BUILD_WITH_MIMALLOC") == "1":
    features.append("mimalloc")


setup(
    rust_extensions=[
        RustExtension(
            "qiskit._accelerate",
            "crates/pyext/Cargo.toml",
            binding=Binding.PyO3,
            debug=rust_debug,
            features=features,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
