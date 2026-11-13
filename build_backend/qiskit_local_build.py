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

"""In-tree PEP 517 build backend for Qiskit.

This is a very thin wrapper around `maturin`.  The purpose of it is to let us continue to use our
system of environment variables to control various things about the build, rather than requiring
people to hard-couple to Maturin's opinionated (and not fantastically documented) PEP 517
configuration.

We should not include any build-critical logic in this build backend; this is only supposed to be
for rewriting custom configurations from Qiskit-specific forms into `maturin`-understood ones.
"""

# The things we export are the PEP 517 / PEP 660 hooks.
__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]

import argparse
import dataclasses
import os
import shlex
import sys
import tomllib
import warnings

import maturin

# PEP 517 guarantees we get run in the root of the source tree.
with open("pyproject.toml", "rb") as fp:
    PYPROJECT_TOML = tomllib.load(fp)


@dataclasses.dataclass(frozen=True, slots=True)
class Feature:
    name: str
    """The name of the Rust feature."""
    env: tuple[str, bool]
    """Which environment sets it, and whether it enables it (``True``) or not (``False``)."""


# The features we might want to override at build time.  The names should match the features of
# `pyext`, while the defaults are taken from the `tool.maturin.features` key of `pyproject.toml`.
KNOWN_FEATURES = [
    Feature("cache_pygates", ("QISKIT_NO_CACHE_GATES", False)),
    Feature("global_mimalloc", ("QISKIT_BUILD_WITH_MIMALLOC", True)),
]


def envvar_bool(envvar: str) -> bool:
    return os.environ.get(envvar, "0") == "1"


def build_profile():
    # Check for a default build profile from the environment. If not present, we also check if
    # `RUST_DEBUG=1` for convenience, since we did that (undocumented) until Qiskit 2.4.
    if (build_profile := os.getenv("QISKIT_BUILD_PROFILE", None)) is None:
        return "dev" if os.getenv("RUST_DEBUG", None) == "1" else None
    profiles = {"debug": "dev", "release": "release"}
    if profile := profiles.get(build_profile.lower()):
        return profile
    warnings.warn(
        f"QISKIT_BUILD_PROFILE set to unknown value '{build_profile}'."
        " Valid values are 'debug' and 'release'."
    )
    return None


def feature_overrides() -> set[str] | None:
    """Convert the feature-setting environment variables into a suitable feature set.

    Returns ``None`` if we resolve to the defaults.  An empty iterable means the feature set needs
    to be explicitly overridden to be empty."""
    # Qiskit's environment variables are additive (or subtractive) from the base configuration,
    # whereas Maturin's `--features`-like flags are overrides.  To make our set of features, we have
    # to start from the defaults and modify, rather than building up a set of additions/removals.
    default_features = frozenset(
        PYPROJECT_TOML.get("tool", {}).get("maturin", {}).get("features", ())
    )
    if unknown := default_features - {feature.name for feature in KNOWN_FEATURES}:
        print(
            f"`tool.maturin.features` contains unknown features: {', '.join(sorted(unknown))}",
            file=sys.stderr,
        )
        sys.exit(1)
    features = set(default_features)
    for feature in KNOWN_FEATURES:
        var, positive = feature.env
        if not envvar_bool(var):
            continue
        if positive:
            features.add(feature.name)
        else:
            features.discard(feature.name)
    return None if features == default_features else features


def update_config_from_environment(
    config_settings: dict[str, str | list[str]] | None = None,
) -> dict[str, str | list[str]] | None:
    """Translate Qiskit-specific environment variables into suitable PEP-517 arguments to the
    `maturin` build system."""
    # Unfortunately Maturin doesn't allow command-line configuration of this, so we force it into
    # the environment.
    if not envvar_bool("QISKIT_ALLOW_AUTOMATIC_RUST_INSTALLATION"):
        os.environ.setdefault("MATURIN_NO_INSTALL_RUST", "1")
    # Maturin moans if `build-backend` isn't `maturin`, but we're deliberately wrapping it.  We
    # unfortunately can't catch direct use of `maturin`.
    os.environ.setdefault("MATURIN_NO_MISSING_BUILD_BACKEND_WARNING", "1")

    config_settings = (config_settings or {}).copy()

    # Best-effort attempt to avoid clobbering manually specified command-line arguments.
    build_args = config_settings.pop("maturin.build-args", config_settings.pop("build-args", ""))
    if isinstance(build_args, str):
        build_args = shlex.split(build_args)

    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument("--profile")
    parser.add_argument("-F", "--features", action="append", default=None)
    parser.add_argument("--all-features", action="store_true")
    (parsed, build_args) = parser.parse_known_args(build_args)

    extra_args = []
    if profile := parsed.profile or build_profile():
        extra_args.append(f"--profile={profile}")

    if parsed.all_features:
        extra_args.append("--all-features")
    elif parsed.features is not None:
        # Both space- and comma-separated arguments are supported.
        features = [feature for arg in parsed.features for feature in arg.replace(",", " ").split()]
        extra_args.append(f"--features={','.join(features)}")
    elif (features := feature_overrides()) is not None:
        # Note that this is a configuration override even if `features` is empty; the case of
        # "unset" is different to "set to empty".
        extra_args.append(f"--features={','.join(sorted(features))}")

    # Ours go at the start, so any `--` present doesn't mess us up.
    build_args = extra_args + build_args
    if build_args:
        config_settings["maturin.build-args"] = build_args
    return config_settings


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
):
    return maturin.build_wheel(
        wheel_directory,
        config_settings=update_config_from_environment(config_settings),
        metadata_directory=metadata_directory,
    )


def build_sdist(
    sdist_directory: str, config_settings: dict[str, str | list[str]] | None = None
) -> str:
    return maturin.build_sdist(
        sdist_directory,
        config_settings=update_config_from_environment(config_settings),
    )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, str | list[str]] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return maturin.build_editable(
        wheel_directory,
        config_settings=update_config_from_environment(config_settings),
        metadata_directory=metadata_directory,
    )


def get_requires_for_build_wheel(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return maturin.get_requires_for_build_wheel(update_config_from_environment(config_settings))


def get_requires_for_build_sdist(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return maturin.get_requires_for_build_sdist(update_config_from_environment(config_settings))


def get_requires_for_build_editable(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return maturin.get_requires_for_build_editable(
        update_config_from_environment(config_settings),
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, str | list[str]] | None = None
) -> str:
    return maturin.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=update_config_from_environment(config_settings),
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: dict[str, str | list[str]] | None = None
) -> str:
    return maturin.prepare_metadata_for_build_editable(
        metadata_directory,
        config_settings=update_config_from_environment(config_settings),
    )
