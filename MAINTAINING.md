# Maintainers Guide

This document defines a *maintainer* as a contributor with merge privileges.
The information detailed here is mostly related to Qiskit releases and other internal processes.


## Package Version

The version of the Qiskit package and crates is set in a few places:

* `qiskit/VERSION.txt` for the Python package and docs
* `Cargo.toml` for the Rust crates
* `crates/cext/cbindgen.toml` for the C header file

In principle, all three version numbers should be the same at all times.
However, the different languages have different conventions about formatting.

### Version-number formatting

Python and Rust use different conventions for pre-release suffixes to the version.
The punctuation (or lack of) separating the main number and the suffix is important.

| Release level      | Python example | Rust or C example |
|--------------------|----------------|-------------------|
| stable             | 2.3.0          | 2.3.0             |
| release candidate  | 2.3.0rc1       | 2.3.0-rc1         |
| beta               | 2.3.0b1        | 2.3.0-beta1       |
| development        | 2.3.0.dev0     | 2.3.0-dev         |

In C the version string is custom and freeform, but we have a test that checks that it matches the Rust one.


### Updating the version number

The package version stored into the repository should be changed as follows:

- on `main`, the package version should almost always have a `dev` suffix and the version number should be the major/minor that is under development on `main`.
  For example, while 2.2.x is the current active release series of Qiskit, the version number on `main` should be `2.3.0.dev0`.

- on a stable branch, the version number should be whatever the most recent release on the stable branch was; it is incremented as part of the release process.
  For example, the `stable/2.3` branch is created from the commit that bumps the version number to `2.3.0rc1`.

The procedure for a new minor-version release, with respect to version numbers is:

1. on `main`, push a PR that bumps the version from `2.2.0.dev0` to `2.2.0rc1` (and moves the loose release notes into `releasenotes/notes/2.2`, and then do the rest of the release process)
2. `qiskit-bot` will create a `stable/2.2` branch from that commit, since that's the one you should tag.
3. on `main`, immediately push a PR that bumps the version to `2.3.0.dev0` to open development on the 2.3 series.

You will need to run `cargo build` as part of a version-bump commit to propagate the changes in `Cargo.toml` to `Cargo.lock`.


## Stable Branch Policy

The stable branch is intended to be a safe source of fixes for high-impact
bugs and security issues that have been fixed on `main` since a
release. When reviewing a stable branch PR, we must balance the risk
of any given patch with the value that it will provide to users of the
stable branch. Only a limited class of changes are appropriate for
inclusion on the stable branch. A large, risky patch for a major issue
might make sense, as might a trivial fix for a fairly obscure error-handling
case. A number of factors must be weighed when considering a
change:

-   The risk of regression: even the tiniest changes carry some risk of
    breaking something, and we really want to avoid regressions on the
    stable branch.
-   The user visibility benefit: are we fixing something that users might
    actually notice, and if so, how important is it?
-   How self-contained the fix is: if it fixes a significant issue but
    also refactors a lot of code, it's probably worth thinking about
    what a less risky fix might look like.
-   Whether the fix is already on `main`: a change must be a backport of
    a change already merged onto `main`, unless the change simply does
    not make sense on `main`.


### Backporting

When a PR tagged with `stable backport potential` is merged, or when a
merged PR is given that tag, the [Mergify bot](https://mergify.com) will
open a PR to the current stable branch.  You can review and merge this PR
like normal.


## Documentation Structure

Qiskit documentation is structured to keep as much of the content as possible within [docstrings](https://peps.python.org/pep-0257/#what-is-a-docstring).
This approach makes it easier to add or update documentation during development, since most of it lives close to the code being modified.
These docstrings are then pulled into [the API Reference section](https://quantum.cloud.ibm.com/docs/api/qiskit) of [quantum.cloud.ibm.com/docs](https://quantum.cloud.ibm.com/docs/).

Refer to [`qiskit_sphinx_theme` docs](https://qiskit.github.io/qiskit_sphinx_theme/apidocs/index.html) for how to create and
write effective API documentation, such as setting up the RST files and docstrings.

If your changes affect non-API content on [quantum.cloud.ibm.com/docs](https://quantum.cloud.ibm.com/docs), you can open an issue (or better yet, a PR) in the [Qiskit/documentation](https://github.com/Qiskit/documentation) repository.
That repository can also be used to suggest or contribute brand-new content beyond updates to the API reference.