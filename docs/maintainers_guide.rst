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
-   Whether the fix is already on ``main``: a change must be a backport of
    a change already merged onto master, unless the change simply does
    not make sense on master.


Backporting
-----------

When a PR tagged with ``stable backport potential`` is merged, or when a
merged PR is given that tag, the `Mergify bot <https://mergify.com>`__ will
open a PR to the current stable branch.  You can review and merge this PR
like normal.


Documentation Structure
=======================

The way documentation is structured in Qiskit is to push as much of the actual
documentation into the docstrings as possible. This makes it easier for
additions and corrections to be made during development, because the majority
of the documentation lives near the code being changed.

Refer to https://qiskit.github.io/qiskit_sphinx_theme/apidocs/index.html for how to create and
write effective API documentation, such as setting up the RST files and docstrings.
