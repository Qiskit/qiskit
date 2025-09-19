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


## The release manager role

The processes around tracking that the correct code goes into a release in a timely manner, and releasing the package when it is time is the responsibility of the **release manager**.

Each release is tracked through a milestone that aggregates all associated issues and PRs.
This milestone should be continuously groomed throughout the release cycle. Additionally, at the release manager's discretion, a GitHub project board may be used to help prioritize tasks and monitor overall progress.
During the release cycle, the release manager is responsible for:

- Tracking the status of the remaining issues.

- Determining when issues/PRs need to be pushed to a later release, and which are high priority and the release should block on.
  Discuss with the team; you don't need to make all the decisions, you're just responsible for making sure
  the decisions are made.

- Making sure there are sufficient coders and reviewers assigned to each PR, and checking up on them to make sure they're progressing.

- If this release is a release candidate for a major or minor release, there is a feature freeze starting two weeks before release day.
  No new public-API-changing PRs (new features or deprecations) can be added to the milestone during this period; they must wait until the next minor release.
  If the API-change is not backwards compatible (like a removal), it needs to wait until the next major, following SemVer.

When we're getting close to release day, **do not** try and race through large and non-blocking PRs.
The release cycle for major and minor releases is regular so these can wait, and if it is a bugfix, we are free to make another patch release quickly.
Non-blocking issues with no associated PR within a day of the release
should be pushed for further release.

## Release process for Qiskit

This section contains the human-intervention parts of the release process for pushing a release and hosting it on PyPI.

The technical process of building, testing and deploying the package is automated.
However, the **release manager** should manually follow these steps:

 1. [Check the milestone state](#1-check-the-milestone-state)
 2. [Audit `Changelog:*` labels](#2-audit-changelog-labels)
 3. [Creating, submitting, and merging a PR to handle release-specific changes as last commit on the milestone](#3-submit-a-prepare-xyz-release-pr)
 4. [Tag the commit from step 3 to trigger the release automation process](#4-tag-the-prepare-xyz-release-commit)
 5. [Make post-release changes in the repository](#5-post-release-actions)

The release process is largely the same for all versions.
However, for convenience, let's put names to two main scenarios for a release:

 * a **_first_ release**: If [`stable/x.y` does not exist in upstream repository](https://github.com/Qiskit/qiskit/branches), you are probably preparing for the first release candidate of a major or minor release, and you are in this scenario.
   Examples of *first* releases are: `0.43.0`, `1.0.0rc1`, and `1.3.0rc1`.
 * a **_follow-up_ release**: If [`stable/x.y` exists in the upstream repository](https://github.com/Qiskit/qiskit/branches), this release is some form of follow-up, probably because there is a previous release candidate or you are doing a patch release.
   Examples of *_follow-up_* releases are: `0.43.3`, `1.2.1`, and `1.3.0rc2`.

> [!NOTE]
> This section assumes that the Qiskit-owned GitHub remote is called `upstream` in your git configuration.


### 1. Check the milestone state

Verify that the milestone is in a suitable place to release:

 - Set the due date for an estimated time for the release, if not set already (for example, in patch release cases).
 - Check for missing items in the milestone. For example, [search for open PRs against stable branches](https://github.com/Qiskit/qiskit/pulls?q=is%3Apr+is%3Aopen+-base%3Amain) and ensure they are labeled with the upcoming release milestone.
 - The day before the release: 
   * all the blocking issues/PR should be merged the day before the release.
   * if it is an rc release, you can leave non-critical bug fixes open, as they could land later, before the final release.
   * non-blocking issues/PR can left open and consider bumping to the next release milestone later. 

### 2. Audit `Changelog:*` labels

> [!NOTE]
> In this section, `(x.y.z)-1` refers to the **the previous version tag**, since the tool needs to consider changes _since_ the version supplied.
> Examples:
> 
>  * For the second release candidate `1.3.0rc2`, `(x.y.z)-1` is previous release candidate `1.3.0rc1`.   
>  * For the first patch release `1.3.1`, `(x.y.z)-1` is the previous stable `1.3.0`.
>  * For the minor release `1.3.0`, `(x.y.z)-1` is the last release candidate `1.3.0rc2`.   

Generate the short-form changelog using [this script](https://github.com/Qiskit/qiskit-bot/blob/master/tools/generate_changelog.py) from [the `qiskit-bot` repository](https://github.com/Qiskit/qiskit-bot):

If this is a **_first_ release** scenario, run `generate_changelog.py` with the following parameters:

```bash
python generate_changelog.py Qiskit/qiskit (x.y.z)-1 -t $MY_GITHUB_API_TOKEN
```

The default behavior of `generate_changelog.py` is to check for changes on the `main` branch of `upstream`. If you are doing a **_follow-up_ release**, run `generate_changelog.py` using the existing `'stable/x.y'` branch:

```bash
python generate_changelog.py Qiskit/qiskit (x.y.z)-1 -b 'stable/x.y' -t $MY_GITHUB_API_TOKEN
```
    
In both scenarios, if there are entries under `Missing changelog entry`, label the PRs (the main and the backport) with the `Changelog:<something>` label and repeat `generate_changelog.py` until all the entries have a changelog label and `Missing changelog` section is not shown.
See [this section](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#changelog-generation) for more details about the available `Changelog:` labels. 

### 3. Submit a "Prepare x.y.z release" PR

Create a PR that will serve as the commit we tag for the release and label it with `Changelog:None`.
The PR is like a regular PR in your fork and submitted like the regular PR process.
This step differs depending on the type of release.

Examples for **_first_ release** PRs:

 * [release-candidate PR for Qiskit 1.3.0rc1](https://github.com/Qiskit/qiskit/pull/13397).
 * [release-candidate PR for Qiskit 2.0.0rc1](https://github.com/Qiskit/qiskit/pull/13953)

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

Once in the `prepare-x.y.z` branch, bump the package version number to `x.y.z` (e.g. `1.4.2` for a patch, or `1.3.0rc2` for a second release candidate).
The places to update are:

* `qiskit/VERSION.txt`: the only line in the file.
* `docs/conf.py`: the variables `version` and `release`.
* `Cargo.toml` (only the file in the repository root and none of the other `**/Cargo.toml`s): the variable `version` and run `cargo build`. Cargo needs a dash before 'rc' dev versions. Therefore, for releasing `x.y.0rc1`, the cargo version should be `x.y.0-rc1`.
* `crates/cext/cbindgen.toml`: the `define QISKIT_VERSION_*` macros. 


#### 3.3 Update Rust dependency

> [!IMPORTANT]
> Skip this step if you are doing a **_follow-up_ release**.

Update any Rust dependencies in the `Cargo.lock` file, keeping the MSRV fixed.
Beware that `cargo`'s dependency resolver will not enforce that dependencies satisfy our `rust-version` support, so you should use our MSRV
of `cargo` to do the update and a trial build.
This will happen by default due to our `rust-toolchain.toml` file, but if you need to temporarily override any toolchain changes you have made locally, do:

```bash
rustup install 1.61  # Install MSRV cargo, if required
cargo +1.61 update   # Update lock file
cargo +1.61 build    # Check build
```

#### 3.4 Add a prelude

Add a release note called `prepare-x.y.z` with only a `prelude` section explaining the release.

The list of features to highlight is usually related to the major themes in the release.
The items in the [Roadmap](https://github.com/Qiskit/qiskit/wiki/Roadmap) can be a good starting point.
The prelude does not usually include code examples or detailed explanations, as it is TL;DR of the rest of the release notes.

Consider the following guidelines:

* For major releases, the format is usually a bullet list of feature highlights, followed by a paragraph or two with the major API breaking changes. [Example of a prelude for the 2.0.0 major release](https://github.com/Qiskit/qiskit/blob/stable/2.0/releasenotes/notes/2.0/prepare-2.0.0-bab067ae93d40bb1.yaml)

* For minor releases, use a bullet list of the major improvements and new features, including any major API deprecations. [Example of a prelude for the 2.1.0 minor release](https://github.com/Qiskit/qiskit/blob/stable/2.1/releasenotes/notes/2.1/prepare-2.1.0-409d24ecbe277062.yaml)

* For patch releases, the prelude can just be something like:

  > Qiskit x.y.z is a small patch release, fixing several bugs found  
  > in the x.y series.


#### 3.5 Prepare the release notes

In case of **_first_ release**, move all the release notes that are loose in `/releasenotes/notes` into a new folder called `/releasenotes/notes/x.y`.
You do not need to fix typos / code / links in the release notes at this stage.
In the busy pre-RC period, your time is likely better spent coordinating and doing final PR reviews.

However, if this is a **_follow-up_ release** keep the loose files in `/releasenotes/notes` and spend some time looking for typos, broken links, and any broken example code blocks in these files.
It's convenient to [build the docs locally](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#building-release-notes-locally) and read through the page, trying the links and code blocks.

#### 3.6 Submit the PR for review

As any other regular PR, commit your changes (don't forget to add the prelude release note), push the branch, and create a PR.

> [!IMPORTANT]
> Pay attention to the base: if you are working in a **_follow-up_ release**, the base is `stable/x.y`.
> Only PR against `main` if you are doing a **_first_ release**.
   
Add the PR you just made to the milestone for this release.
This is the last PR that should merge from the milestone.
This PR undergoes the regular review process - use the reviewers to help with checking all the release notes if you need to.

### 4. Tag the "Prepare x.y.z release" commit

> [!WARNING]
> To push the tag that triggers the final deployment of a release, you need to have at least `maintain` permissions on the repository.
> All of the preparation before that can be done by anyone, subject to the standard PR review processes; the `maintain` permission is only required to push a tag to the Qiskit git repository.

Once the PR from the previous section is merged, the release manager tags the commit of that PR.  The tag should have:

- A tag name exactly equal to the version number
- A tag message that says "Qiskit x.y.z"
- Ideally, [sign your tagging using GPG](https://docs.github.com/authentication/managing-commit-signature-verification/signing-tags)

#### 4.1 Create the tag locally

The following are the recommended steps for tagging the "Prepare x.y.z release" commit:

1. Sync with the Qiskit-owned remote: `git fetch upstream`
2. Make sure your commit from previous section is `HEAD` in the stable branch:  `git show upstream/stable/x.y` (in case of **_follow-up_ release**) or `git show upstream/main` (in case of **_first_ release**)
3. Tag with a signature and message: `git tag -s -m "Qiskit x.y.z" x.y.z upstream/stable/x.y` (in case of **_follow-up_ release**) or `git tag -s -m "Qiskit x.y.z" x.y.z upstream/main` (in case of **_first_ release**)

>[!TIP]
> Signing the tag is optional but highly recommended. Omit `-s` in `git tag` if you are not signing the tag.

For example, here is the workflow for creating the tag for the 2.0.3 patch release, immediately after the "Prepare 2.0.3 release" PR has been merged:

```bash
git fetch upstream
git show upstream/stable/2.0  # Verify this is the release PR.
git tag -s -m "Qiskit 2.0.3" 2.0.3 upstream/stable/2.0
```

#### 4.2 Verify the tag

Double-check that the tag you have just created has exactly the correct name,
and points to exactly the correct commit:

```bash
git show x.y.z
```

<pre>
<code>
tag <b>x.y.z</b>
Tagger: ...
Date:   ...

Qiskit <b>x.y.z</b>
-----BEGIN PGP SIGNATURE-----
....
-----END PGP SIGNATURE-----

commit ... (tag: <b>x.y.z</b>, upstream/<b>stable/x.y</b>, ...)
Author: ...
Date:   ...

Prepare <b>x.y.z</b> release (#....)
</code>
</pre>

Check that the bold parts are correct.

For example, for the `2.0.3` release:

```bash
git show 2.0.3
```

```text
tag 2.0.3
Tagger: Matthew Treinish <mtreinish@k***r.org>
Date:   Tue Jun 17 08:30:03 2025 -0400

Qiskit 2.0.3
-----BEGIN PGP SIGNATURE-----
....
-----END PGP SIGNATURE-----

commit 19eeb418...14482636a (tag: 2.0.3, upstream/stable/2.0, ...)
Author: Matthew Treinish <mtreinish@k***r.org>
Date:   Tue Jun 17 07:19:06 2025 -0400

Prepare 2.0.3 release (#14626)
```

Note that the tagged commit is precisely the "Prepare 2.0.3 release" commit.


#### 4.3 Push the tag to Qiskit remote

> [!WARNING]
> This step triggers the release.

Push the tag to the Qiskit remote with `git push upstream x.y.z`.
Following the previous example:

```bash
git push upstream 2.0.3
```

At this point, the release-automation process takes over.
[`qiskit-bot`](https://github.com/Qiskit/qiskit-bot) will populate a GitHub release with the new tag and the short-form changelog seen in [step 2](#2.-Verify-`Changelog`-labels).


If this is a **_first_ release**, `qiskit-bot` will create a new `stable/x.y` branch for this series.
The GitHub Actions CD pipelines will build the sdist, and the wheels for all
Python / OS / architecture combinations, and push them to PyPI using the
encrypted credentials in this repository.

#### 4.4 Get approval for pushing to PyPI

The first GitHub Actions CD stage builds the [Tier 1](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit#operating-system-support) and takes between 1.5 and 2 hours.
Once it finishes, the wheels can be pushed to PyPI by [the deploy workflow](https://github.com/Qiskit/qiskit/blob/main/.github/workflows/wheels.yml).
For that, an active approval by somebody (excluding the release manager) from the [release team](https://github.com/orgs/Qiskit/teams/terra-release) needs to be performed.

> TODO: Explain what the approver needs to check for.

Traditionally, once Tier 1 is live on PyPI, the post-release actions in step 5 can start, including the announcements.
The rest of the tiers might take even longer and they also need to be approved.


### 5. Post-release actions

#### 5.1 Announce the release on Slack

Post a message in the relevant Slack channels:

 * IBM internal channels: all the releases.
 * [Qiskit organization](https://qiskit.enterprise.slack.com/)
    - Roadmap announcement `#roadmap-announcements`: Especially release candidates. All other releases can be posted in the thread of the release candidate announcement.
    - General channel `#general` and Qiskit developer `#qiskit-dev`: All the stable major and minor releases.

Examples for announcements:

For a release candidate:
> :qiskit-new: Qiskit x.y.zrc1 is now live on Github (link) and PyPI(link)! :rocket:
>
> As this is a pre-release pip will not install it automatically, you will have to manually specify the version with: `pip install "qiskit==x.y.zrc1"`
> The x.y.z final release is planned for X weeks from now. If you encounter any issues with the release candidate, please [file an issue](https://github.com/Qiskit/qiskit/issues/new/choose) so we can address them before the final release.

For minor releases:
> :qiskit-new: **Qiskit x.y is out!**
>
> * Fully backwards compatible with x.0. As always, following [Semantic Versioning](https://qisk.it/semver)
> * A technical release summary will be published on [the IBM blog](https://www.ibm.com/quantum/blog) in about a week
>
> or
>
> * Here is technical release summary (link to the blog post)
> * Take a look to the release notes (link)
> Don't forget that the Qiskit vX series has bug fixing support until XXth, XXXX and security support until YYth, YYYY.

For patch releases:
> :qiskit-new: **Qiskit x.y.z has been released!**
> This is a minor bugfix release for Qiskit x.y. You can find it on pypi (link) and in our GitHub releases (link).

#### 5.2 Update the `main` branch with the next release

> [!IMPORTANT]
> Skip this step if your PR from section Releasing the package - step 3 was to the  `main` branch

Make a PR to the `main` branch that sets the version number to the _next_ minor (e.g. `2.2.0` if you have just released `2.1.0rc1`).
The places to update are:

* `qiskit/VERSION.txt`: the only line in the file.
* `docs/conf.py`: the variables `version` and `release`.
* `Cargo.toml` (only the file in the repository root and none of the other `**/Cargo.toml`s): the variable `version` and run `cargo build`.
* `crates/cext/cbindgen.toml`: the `#define QISKIT_VERSION_*` macros.
* `.mergify.yml`: change the `backport` action to target the new stable branch that `qiskit-bot` created as part of the release (see [*Backporting* section](#Backporting) for details on Mergify).

This opens the `main` branch for feature development for the next release.

Examples for post-release PR:
 * [following 2.1.0rc1, a PR preparing `main` for 2.2.0](https://github.com/Qiskit/qiskit/pull/14546)
 * [following 2.0.0rc1, a PR preparing `main` for 2.1.0](https://github.com/Qiskit/qiskit/pull/13982)

#### 5.3 Create a milestone for the next patch release

> [!IMPORTANT]
> Skip this step if you released a release candidate.

Once a package is out there, it has support for certain period of time.
As such, there are potential patch releases coming and it is handy to have [a milestone](https://github.com/Qiskit/qiskit/milestones/) ready for that.
If you have any estimated or time plan for this future patch release, consider adding it.

#### 5.4 Update the roadmap

Go to the [roadmap wiki](https://github.com/Qiskit/qiskit/wiki/Roadmap) and update it:

- If a version reached end-of-life, move the full version section to the _unmaintained versions_ fold
- If it was a patch release, update the milestone links.
- If new release notes are available, link them.
- If there were items that didn't make it into the release, move them to the next minor/major so they can be considered again.
