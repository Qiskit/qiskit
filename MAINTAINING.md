# Maintainers Guide

This document defines a *maintainer* as a contributor with merge privileges.
The information detailed here is mostly related to Qiskit releases and other internal processes.


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

The way documentation is structured in Qiskit is to push as much of the actual
documentation into the docstrings as possible. This makes it easier for
additions and corrections to be made during development, because the majority
of the documentation lives near the code being changed. These docstrings are then pulled into
the API Reference section of https://quantum.cloud.ibm.com/docs.

Refer to https://qiskit.github.io/qiskit_sphinx_theme/apidocs/index.html for how to create and
write effective API documentation, such as setting up the RST files and docstrings.

If changes you are making affect non-API reference content in https://quantum.cloud.ibm.com/docs
you can open an issue (or better yet a PR) to update the relevant page in https://github.com/Qiskit/documentation.
You can also use this repo to suggest or contribute brand new content beyond updates to the API reference.


# Release process for Qiskit

This chapter contains the human-intervention parts of the release process for pushing a release and hosting it on PyPI.

The technical process of the packages being built, tested and deployed is automated.
The processes around tracking that the correct code goes into a release in a timely manner, and updating the version numbers of the package when it is time is the responsibility of the **release manager**.

> [!Note]
> This chapter assumes that the Qiskit-owned GitHub remote is called `upstream` in your git configuration.


## Permissions

To push the tag that triggers the final deployment of a release, you need to have at least `maintain` permissions on the relevant repository.
All of the preparation before that can be done by anybody, subject to the standard PR review processes; the `maintain` permission is only required to push a tag to the Qiskit git repository.


## Preparing for release day

There should already be a milestone open that tracks all issues and PRs that have been assigned to this release, and this should be groomed over the course of the entire release cycle.
In the last week before release, the release manager is responsible for:

- tracking the status of the remaining issues.

- determining when issues/PRs need to be pushed to a later release, and which are high priority and the release should block on.
  Discuss with the team; you don't need to make all the decisions, you are just responsible for making sure the decisions are made.

- making sure there are sufficient coders and reviewers assigned to each PR, and checking up on them to make sure they're progressing.

- if this release is a release candidate for a minor release, there is a feature freeze starting two weeks before release day.
  No new public-API-changing PRs (features, deprecations or removals) can be added to the milestone during this period; they must wait until the next minor release.

When we're getting close to release day, don't try and race through large and non-blocking PRs.
The release cycle for minor releases is regular so these can wait, and if it's a bugfix, we're free to make another patch release quickly.
Non-blocking issues with no associated PR within a day or two of the release should be pushed for a later release.

The last PR to clear a milestone should be the "Prepare x.y.z release" PR discussed in the sections below.


## Releasing the package

The release process is largely the same for all versions.
However, for convenience, let's put names to two main scenarios for a release:

 * a **_first_ release**: If [`stable/x.y` does not exist in upstream repository](https://github.com/Qiskit/qiskit/branches), you are probably preparing for the first release candidate of a major or minor release, and you are in this scenario.
   Examples of *first* releases are: `0.43.0`, `1.0.0rc1`, and `1.3.0rc1`.
 * a **_follow-up_ release**: If [`stable/x.y` exists in the upstream repository](https://github.com/Qiskit/qiskit/branches), this release is some form of follow-up, probably because there is a previous release candidate or you are doing a patch release.
   Examples of *_follow-up_* releases are: `0.43.3`, `1.2.1`, and `1.3.0rc2`.


### 1. Check the milestone state

Verify that the milestone is in a suitable place to release.

> TODO: explain what it means to be "in a suitable place".


### 2. Verify `Changelog` labels

You might want to preview the short-form changelog using [this script belonging to `qiskit-bot`](https://github.com/Qiskit/qiskit-bot/blob/master/tools/generate_changelog.py).

>![Note]
> In this section, `(x.y.z)-1` refers to the **the previous version tag**, since the tool needs to consider changes _since_ the version supplied.
> Examples:
> 
>  * For the second release candidate `1.3.0rc2`, `(x.y.z)-1` is `1.3.0rc1`.   
> > TODO: add more examples.


If this is a **_first_ release** scenario, run `generate_changelog.py` with the following parameters:

```bash
python generate_changelog.py Qiskit/qiskit (x.y.z)-1 -t $MY_GITHUB_API_TOKEN
```

The default behavior of `generate_changelog.py` is to check for changes on the `main` branch of `upstream`. If you are doing a **_follow-up_ release**, run `generate_changelog.py` using the existing `'stable/x.y'` branch:

```bash
python generate_changelog.py Qiskit/qiskit x.y.(z-1) -b 'stable/x.y' -t $MY_GITHUB_API_TOKEN
```
       
In both scenarios, if there are entries under `Missing changelog entry`, label the PRs (the main and the backport) with the `Changelog:<something>` label and repeat `generate_changelog.py` until all the entries have a changelog label and `Missing changelog` section is not shown.


### 3. Prepare a "Prepare x.y.z release" PR

Prepare a PR that will be used as the commit to tag the release.
The PR is like a regular PR in your fork and submitted like the regular PR process.

Example for **_first_ release** PR:

 * [release-candidate PR for Qiskit 1.3.0rc1](https://github.com/Qiskit/qiskit/pull/13397).

Examples for **_follow-up_ release** PRs:

 * [patch-release PR for Qiskit 2.0.1](https://github.com/Qiskit/qiskit/pull/14339).
 * [follow-up release-candidate PR for Qiskit 1.3.0rc2](https://github.com/Qiskit/qiskit/pull/13466/).  

#### 3.1 Create the `prepare-x.y.z` branch


If this is a **_first_ release**, `stable/x.y` does not exist and you should create the branch out of `main`:

```bash
git checkout -b prepare-x.y.0rc1 upstream/main
```

If you are doing a **_follow-up_ release**, make a PR out of the stable branch:

```bash
git fetch upstream
git checkout -b prepare-x.y.z upstream/stable/x.y
```

In both situations, your active branch now is `prepare-x.y.z`.

#### 3.2 Bump version numbers

Once in the `prepare-x.y.z` branch, bump the package version number to `x.y.z` (e.g. `1.4.2` for a patch, or `1.3.0rc2` for a second release candidate). The places to update are:

* `qiskit/VERSION.txt`: the only line in the file.
* `docs/conf.py`: the variables `version` and `release`.
* `Cargo.toml` (only the file in the repository root and none of the other `**/Cargo.toml`s): the variable `version` and run `cargo build`. Cargo doesn't allow 'rc' dev versions. Therefore, for releasing `x.y.0rc1`, the cargo version should be `x.y.0`.

#### 3.3 Update Rust dependency

> [!IMPORTANT]
> Skip this step if you are doing a **_follow-up_ release**.

Update any Rust dependencies in the `Cargo.lock` file, keeping the MSRV
fixed.  Beware that `cargo`'s dependency resolver will not enforce that
dependencies satisfy our `rust-version` support, so you should use our MSRV
of `cargo` to do the update and a trial build.  This will happen by
default due to our `rust-toolchain.toml` file, but if you need to
temporarily override any toolchain changes you have made locally, do:

```bash
rustup install 1.61  # Install MSRV cargo, if required
cargo +1.61 update   # Update lock file
cargo +1.61 build    # Check build
```

#### 3.4 Add a prelude

Add a release note called `prepare-x.y.z` with only a `prelude` section explaining the release.

> TODO: add more examples

For patch releases, this can be something like

> Qiskit x.y.z is a small patch release, fixing several bugs found
> in the x.y series.

#### 3.5 Prepare the release notes

In case of **_first_ release**, move all the release notes that are loose in `/releasenotes/notes` into a new folder called `/releasenotes/notes/x.y`.
You do not need to fix typos / code / links in the release notes at this stage.
In the busy pre-RC period, your time is likely better spent coordinating and doing final PR reviews.

However, if this is a **_follow-up_ release** keep the loose files in `/releasenotes/notes` and spend some time looking for typos, broken links, and any broken example code blocks in these files.
It's convenient to [build the docs locally](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#building-release-notes-locally) and read through the page, trying the links and code blocks.

#### 3.6 Submit the PR for revision

As a regular PR, commit your changes (don't forget to add the prelude release note), push the branch, and create a PR.

> [!IMPORTANT]
> Pay attention to the base: if you are working in a **_follow-up_ release**, the base is `stable/x.y`.
> Only PR against `main` if you are doing a **_first_ release**.
   
Add the PR you just made to the milestone for this release.
This is the last PR that should merge from the milestone.
This PR undergoes the regular review process - use the reviewers to help with checking all the release notes if you need to.

### 4. Tag the "Prepare x.y.z release" commit

Once the PR from the previous section is merged, the release manager (you) tags the commit of that PR.  The tag should have:

- a tag name exactly equal to the version number
- a tag message that says "Qiskit $version_number"
- ideally, your PGP signature.  Ask around the team (e.g. Matthew or Jake) if
 you want some help getting git commit signing set up.

#### 4.1 Create the tag locally

Example workflow for creating the tag for the 0.24.2 patch release, immediately after the "prepare 0.24.2" PR has been merged:

```bash
git fetch origin
git show origin/stable/0.24  # Verify this is the release PR.
git tag -s -m "Qiskit 0.24.2" 0.24.2 origin/stable/0.24
```

Omit `-s` if you are not signing the tag.


#### 4.2 Verify the tag

Double-check that the tag you have just created has exactly the correct name, and points to exactly the correct commit.  For example:

```bash
git show 0.23.2
```
```text
tag 0.23.2
Tagger: John Doe <jdoe@gmail.com>
Date:   Thu Feb 23 14:26:10 2023 +0100

Qiskit Terra 0.23.2
-----BEGIN PGP SIGNATURE-----
[snip]
-----END PGP SIGNATURE-----

commit 09f904a03c056abb5ed80030e4d1f75108943502 (tag: 0.23.2)
Author: John Doe <jdoe@gmail.com>
Date:   Thu Feb 23 12:18:44 2023 +0100

   Prepare 0.23.2 (#9643)
```

Note that the tagged commit is precisely the "Prepare 0.23.2" commit.

#### 4.3 Push the tag to Qiskit remote

> [!IMPORTANT]
> This step triggers the release

Push the tag to the Qiskit remote.  For example:

```bash
git push origin 0.24.2
```

At this point, the release-automation process takes over.
`qiskit-bot` will populate a GitHub release with the new tag and the short-form changelog seen in [step 2](2.-Verify-`Changelog`-labels).

If this is a **_first_ release**, `qiskit-bot` will create a new `stable/x.y` branch for this series.
The GitHub Actions CD pipelines will build the sdist, and the wheels for all
Python / OS / architecture combinations, and push them to PyPI using the
encrypted credentials in this repository.

## Post release

### 1. Announce the release on Slack


Post a message in the relevant Slack channels:

 * IBM internal channels: all the releases.
 * [Qiskit organization](https://qiskit.enterprise.slack.com/)
    - General `#general`: only if it is a major or a minor stable release.
    - Roadmap announcement `#roadmap-announcements`: Especially release candidates. All other releases (except patch releases) can be posted in the thread of the release candidate announcement.
    - General channel `#general` and Qiskit developer `#qiskit-dev`: All the stable major and minor releases.

> TODO: example/s of the post.

### 2. Update the `main` branch with the next release

> [!IMPORTANT]
> Skip this step if your PR from section Releasing the package - step 3 was to the  `main` branch

Submit a PR to the `main` branch that sets the version number to the _next_ minor (e.g. `0.26.0` if you have just released `0.25.0rc1` [TODO: update the example]). 

* `qiskit/VERSION.txt`: the only line in the file.
* `docs/conf.py`: the variables `version` and `release`.
* `Cargo.toml` (only the file in the repository root and none of the other `**/Cargo.toml`s): the variable `version` and run `cargo build`.
* `.mergify.yml`: change the `backport` action to target the new stable branch that `qiskit-bot` created as part of the release - Mergify handles backports in the Qiskit repository.

This opens the `main` branch for feature development for the next release.

Examples for post-release PR:
 * [following 0.24.0rc1, a PR preparing `main` for 0.25.0](https://github.com/Qiskit/qiskit-terra/pull/10005).

> [TODO] add more modern examples.

### 3. Create a milestone for the next patch release

> [!IMPORTANT]
> Skip this step if you released a release candidate.

Once a package is out there, it has support for certain period of time.
As such, there are potential patch releases coming and it is handy to have [a milestone](https://github.com/Qiskit/qiskit/milestones/) ready for that.
If you have any estimated or time plan for this future patch release, consider adding it.

### 4. Update the roadmap

Go to the [roadmap wiki](https://github.com/Qiskit/qiskit/wiki/Roadmap) to update it:

- If a version reached end-of-life move the full version section to the _unmaintained versions_ fold
- If it was a patch release, update the milestone links.
- If new release notes are available, link them.
- If there were items that didn't make it into the release, move them to the next minor/major so they can be considered again.
