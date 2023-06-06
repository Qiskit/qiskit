
#################
Maintainers Guide
#################

This document defines a *maintainer* as a contributor with merge privileges.
The information detailed here is mostly related to Qiskit releases and other internal processes.

.. _stable_branch_policy:

Stable Branch Policy
====================

The stable branch is intended to be a safe source of fixes for high-impact
bugs and security issues that have been fixed on master since a
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
-   Whether the fix is already on master: a change must be a backport of
    a change already merged onto master, unless the change simply does
    not make sense on master.



Backporting procedure:
----------------------

When backporting a patch from master to stable, we want to keep a
reference to the change on master. When you create the branch for the
stable PR, use::

    $ git cherry-pick -x $master_commit_id

However, this only works for small self-contained patches from master.
If you need to backport a subset of a larger commit (from a squashed PR,
for example) from master, do this manually. In these cases, add::

    Backported from: #master pr number

so that we can track the source of the change subset, even if
a strict cherry-pick doesn\'t make sense.

If the patch you're proposing will not cherry-pick cleanly, you can help
by resolving the conflicts yourself and proposing the resulting patch.
Please keep Conflicts lines in the commit message to help review of the
stable patch.



Backport labels
---------------

Bugs or PRs tagged with ``stable backport potential`` are bugs
that apply to the stable release too and may be suitable for
backporting once a fix lands in master. Once the backport has been
proposed, the tag should be removed.

Include ``[Stable]`` in the title of the PR against the stable branch,
as a sign that setting the target branch as stable was not
a mistake. Also, reference to the PR number in master that you are
porting.

.. _versioning_strategy:

*****************
Qiskit Versioning
*****************

The Qiskit project is made up of several elements each performing different
functionality. Each is independently useful and can be used on their own,
but for convenience we provide this repository and meta-package to provide
a single entrypoint to install all the elements at once. This is to simplify
the install process and provide a unified interface to end users. However,
because each Qiskit element has its own releases and versions, some care is
needed when dealing with versions between the different repositories. This
document outlines the guidelines for dealing with versions and releases of
both Qiskit elements and the meta-package.

For the rest of this guide the standard Semantic Versioning nomenclature will
be used of: ``Major.Minor.Patch`` to refer to the different components of a
version number. For example, if the version number was ``0.7.1``, then the major
version is ``0``, the minor version ``7``, and the patch version ``1``.


Meta-package Version
====================

The Qiskit meta-package version is an independent value that is determined by
the releases of each of the elements being tracked. Each time we push a release
to a tracked component (or add an element) the meta-package requirements, and
version will need to be updated and a new release published. The timing should
be coordinated with the release of elements to ensure that the meta-package
releases track with element releases.

Adding New Tracked Elements
---------------------------

When a new Qiskit element is being added to the meta-package requirements, we
need to increase the **Minor** version of the meta-package.

For example, if the meta-package is tracking 2 elements ``qiskit-aer`` and
``qiskit-terra`` and its version is ``0.7.4``. Then we release a new element
``qiskit-new`` that we intend to also have included in the meta-package. When
we add the new element to the meta-package we increase the version to
``0.8.0``.


Patch Version Increases
-----------------------

When any Qiskit element that is being already tracked by the meta-package
releases a patch version to fix bugs in a release, we need also bump the
requirement in the ``setup.py`` and then increase the patch version of the
meta-package.

For example, if the meta-package is tracking 2 elements ``qiskit-terra==0.8.1``
and ``qiskit-aer==0.2.1``, with the current version
``0.9.6``. When qiskit-terra release a new patch version to fix a bug ``0.8.2``
the meta-package will also need to increase its patch version and release,
becoming ``0.9.7``.

Additionally, there are occasionally packaging or other bugs in the
meta-package itself that need to be fixed by pushing new releases. When those
are encountered we should increase the patch version to differentiate it from
the broken release. Do **not** delete the broken or any old releases from pypi
in any situation, instead just increase the patch version and upload a new
release.

Minor Version Increases
-----------------------

Besides when adding a new element to the meta-package, the minor version of the
meta-package should also be increased anytime a minor version is increased in
a tracked element.

For example, if the meta-package is tracking 2 elements ``qiskit-terra==0.7.0``
and ``qiskit-aer==0.1.1`` and the current version is ``0.7.5``. When the
``qiskit-aer`` element releases ``0.2.0`` then we need to increase the
meta-package version to be ``0.8.0`` to correspond to the new release.

Major Version Increases
-----------------------

The major version is different from the other version number components. Unlike
the other version number components, which are updated in lock step with each
tracked element, the major version is only increased when all tracked versions
are bumped (at least before ``1.0.0``). Right now, all the elements still have
a major version number component of ``0``, and until each tracked element in the
meta-repository is marked as stable by bumping the major version to be ``>=1``,
then the meta-package version should not increase the major version.

The behavior of the major version number component tracking after when all the
elements are at >=1.0.0 has not been decided yet.

Optional Extras
---------------

In addition to the tracked elements, there are additional packages built
on top of Qiskit which are developed in tandem with Qiskit, for example, the
application repositories like qiskit-optimization. For convienence
these packages are tracked by the Qiskit metapackage as optional extras that
can be installed with Qiskit. Releases of these optional downstream projects
do not trigger a metapackage release as they are unpinned and do not affect the
metapackage version. If there is a compatibility issue between Qiskit and these
downstream optional dependencies and the minimum version needs to be adjusted
in a standalone release, this will only be done as a patch version release as
it's a packaging bugfix.

Qiskit Element Requirement Tracking
===================================

While not strictly related to the meta-package and Qiskit versioning, how we
track the element versions in the meta-package's requirements list is
important. Each element listed in the ``setup.py`` should be pinned to a single
version. This means that each version of Qiskit should only install a single
version for each tracked element. For example, the requirements list at any
given point should look something like::

  requirements = [
      "qiskit_terra==0.7.0",
      "qiskit-aer==0.1.1",
  ]

This is to aid in debugging, but also make tracking the versions across
multiple elements more transparent.

Documentation Structure
=======================

The way documentation is structured in Qiskit is to push as much of the actual
documentation into the docstrings as possible. This makes it easier for
additions and corrections to be made during development, because the majority
of the documentation lives near the code being changed.

Refer to https://qiskit.github.io/qiskit_sphinx_theme/apidocs/index.html for how to create and
write effective API documentation, such as setting up the RST files and docstrings.

qiskit-metapackage repository
-----------------------------

Historically, Qiskit Ecosystem projects were hosted at https://qiskit.org/documentation/.
Those projects now live under https://qiskit.org/ecosystem and https://qiskit.org/documentation
is only for core Qiskit (aka Terra).

While we finish this migration, documentation for Qiskit is still built in
https://github.com/Qiskit/qiskit-metapackage. The metapackage will Git clone Terra to pull in its
API documentation, using whatever version of Terra is pinned in the ``setup.py``. This
means that fixes for incorrect API documentation will need to be
included in a new release. Documentation fixes are valid backports for a stable
patch release per the stable branch policy (see :ref:`stable_branch_policy`).

This setup is temporary and we are migrating the metapackage documentation infrastructure to live
in Terra. See https://github.com/Qiskit/RFCs/issues/41 to track the migration.
