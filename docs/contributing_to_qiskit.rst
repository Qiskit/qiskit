
######################
Contributing to Qiskit
######################

Qiskit is an open-source project committed to bringing quantum computing to
people of all backgrounds. This page describes how you can join the Qiskit
community in this goal.

.. _where_things_are:

****************
Where Things Are
****************

The code for Qiskit is located in the `Qiskit GitHub organization <https://github.com/Qiskit>`__,
where you can find the individual projects that make up Qiskit, including

* `Qiskit Terra <https://github.com/Qiskit/qiskit-terra>`__
* `Qiskit Aer <https://github.com/Qiskit/qiskit-aer>`__
* `Qiskit Ignis <https://github.com/Qiskit/qiskit-ignis>`__
* `Qiskit Aqua <https://github.com/Qiskit/qiskit-aqua>`__
* `Qiskit IBMQ Provider <https://github.com/Qiskit/qiskit-ibmq-provider>`__
* `Qiskit Tutorials <https://github.com/Qiskit/qiskit-tutorials>`__
* `Qiskit API Documentation <https://github.com/Qiskit/qiskit/tree/master/docs>`__


****************
Getting Started
****************

Learn how members of the Qiskit community

* `Relate to one another <https://github.com/Qiskit/qiskit/blob/master/CODE_OF_CONDUCT.md>`__
* `Discuss ideas <https://qiskit.slack.com/>`__
* `Get help when we're stuck <https://quantumcomputing.stackexchange.com/questions/tagged/qiskit>`__
* `Stay informed of news in the community <https://medium.com/qiskit>`__
* `Keep a consistent style <https://www.python.org/dev/peps/pep-0008>`__
* :ref:`Build Qiskit packages from source <install_install_from_source_label>`



******************************************
Reporting Bugs and Requesting Enhancements
******************************************

When you encounter a problem, please open an issue in the
appropriate element's issue tracker:


=========================== =============================================
Element                     Issue Tracker
=========================== =============================================
qiskit-terra                https://github.com/Qiskit/qiskit-terra/issues
qiskit-aer                  https://github.com/Qiskit/qiskit-aer/issues
qiskit-ignis                https://github.com/Qiskit/qiskit-ignis/issues
qiskit-aqua                 https://github.com/Qiskit/qiskit-aqua/issues
Docs or Qiskit Meta-package https://github.com/Qiskit/qiskit/issues
=========================== =============================================

If you have an idea for a new feature, please open an **Enhancement** issue in
the appropriate element's issue tracker.
Opening an issue starts a discussion with the team about your idea, how it
fits in with the project, how it can be implemented, etc.


*****************
Contributing Code
*****************



Style Guide
===========

To enforce a consistent code style in the project, we use `Pylint
<https://www.pylint.org>`__ and `pycodesytle
<https://pycodestyle.readthedocs.io/en/latest/>`__ to verify that code
contributions conform to and respect the project's style guide. To verify that
your changes conform to the style guide, run: ``tox -elint``



Contributor License Agreement
=============================

Before you can submit any code, all contributors must sign a
contributor license agreement (CLA). By signing a CLA, you're attesting
that you are the author of the contribution, and that you're freely
contributing it under the terms of the Apache-2.0 license.

When you contribute to the Qiskit project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The `individual CLA <https://qiskit.org/license/qiskit-cla.pdf>`__
document is available for review as a PDF.

.. note::
   If your contribution is part of your employment or your contribution
   is the property of your employer, then you will more than likely need to sign a
   `corporate CLA <https://qiskit.org/license/qiskit-corporate-cla.pdf>`__ too and
   email it to us at <qiskit@us.ibm.com>.



Pull Requests
=============

We use `GitHub pull requests
<https://help.github.com/articles/about-pull-requests>`__ to accept
contributions.

While not required, opening a new issue about the bug you're fixing or the
feature you're working on before you open a pull request is an important step
in starting a discussion with the community about your work. The issue gives us
a place to talk about the idea and how we can work together to implement it in
the code. It also lets the community know what you're working on, and if you
need help, you can reference the issue when discussing it with other community
and team members.

If you've written some code but need help finishing it, want to get initial
feedback on it prior to finishing it, or want to share it and discuss prior
to finishing the implementation, you can open a *Work in Progress* pull request.
When you create the pull request, prepend the title with the **\[WIP\]** tag (for
Work In Progress). This will indicate to reviewers that the code in
the PR isn't in its final state and will change. It also means that we will
not merge the commit until it is finished. You or a reviewer can remove the
[WIP] tag when the code is ready to be fully reviewed for merging.



Code Review
===========

Code review is done in the open and is open to anyone. While only maintainers have
access to merge commits, community feedback on pull requests is extremely valuable.
It is also a good mechanism to learn about the code base. You can
view a list of all open pull requests here:

=========================== =============================================
Element                     Pull Requests
=========================== =============================================
qiskit-terra                https://github.com/Qiskit/qiskit-terra/pulls
qiskit-aer                  https://github.com/Qiskit/qiskit-aer/pulls
qiskit-ignis                https://github.com/Qiskit/qiskit-ignis/pulls
qiskit-aqua                 https://github.com/Qiskit/qiskit-aqua/pulls
Docs or Qiskit Meta-package https://github.com/Qiskit/qiskit/pulls
=========================== =============================================




Commit Messages
===============

The content of the commit message describing a change is just as important as the
change itself. The commit message provides the context for
not only code review but also the change history in the git log. A detailed
commit message will make it easier for your code to be reviewed, and will also provide
context to the change when someone looks at it in the future. When writing a commit
message, remember these important details:

Do not assume the reviewer understands what the original problem was.
   When reading an issue, after a number of back & forth comments, it is often
   clear what the root cause problem is. The commit message should have a clear
   statement as to what the original problem is. The bug is merely interesting
   historical background on *how* the problem was identified. It should be
   possible to review a proposed patch for correctness from the commit message,
   without needing to read the bug ticket.

Do not assume the code is self-evident/self-documenting.
   What is self-evident to one person, might not be clear to another person. Always
   document what the original problem was and how it is being fixed, for any change
   except the most obvious typos, or whitespace-only commits.

Describe why a change is being made.
   A common mistake is only to document how the code has been written, without
   describing *why* the developer chose to do it that way. Certainly, you should describe
   the overall code structure, particularly for large changes, but more importantly,
   be sure to describe the intent/motivation behind the changes.

Read the commit message to see if it hints at improved code structure.
   Often when describing a large commit message, it becomes obvious that a commit
   should have been split into two or more parts. Don't be afraid to go back
   and rebase the change to split it up into separate pull requests.

Ensure sufficient information to decide whether to review.
   When GitHub sends out email alerts for new pull request submissions, there is
   minimal information included - usually just the commit message and the list of
   files changes. Because of the high volume of patches, a commit message must
   contain sufficient information for potential reviewers to find the patch that
   they need to review.

The first commit line is the most important.
   In Git commits, the first line of the commit message has special significance.
   It is used as the default pull request title, email notification subject line,
   git annotate messages, gitk viewer annotations, merge commit messages, and many
   more places where space is at a premium. As well as summarizing the change
   itself, it should take care to detail what part of the code is affected.

   In addition, the first line of the commit message becomes an entry in the
   generated changelog if the PR is tagged as being included in the changelog.
   It is critically important that you write clear and succinct summary lines.

Describe any limitations of the current code.
   If the code being changed still has future scope for improvements, or any known
   limitations, mention these in the commit message. This demonstrates to the
   reviewer that the broader picture has been considered, and what tradeoffs have
   been done in terms of short-term goals versus long-term wishes.

Include references to issues.
   If the commit fixes are related to an issue, make sure you annotate that in
   the commit message. Use the syntax::

       Fixes #1234

   if it fixes the issue (GitHub will close the issue when the PR merges).

The main rule to follow is:

The commit message must contain all the information required to fully
understand and review the patch for correctness. Less is not more.



Documenting Your Code
=====================

If you make a change to an element, make sure you update the associated
*docstrings* and parts of the documentation under ``docs/apidocs`` in the
corresponding repo. To locally build the element-specific
documentation, run ``tox -edocs`` to compile and build the
documentation locally and save the output to ``docs/_build/html``.
Additionally, the Docs CI job on azure pipelines will run this and host a zip
file of the output that you can download and view locally.

If you have an issue with the `combined documentation <https://qiskit.org/documentation/>`__
that is maintained in the `Qiskit/qiskit repo <https://github.com/Qiskit/qiskit>`__,
you can open a `documentation issue <https://github.com/Qiskit/qiskit/issues/new/choose>`__
if you see doc bugs, have a new feature that needs to be documented, or think
that material could be added to the existing docs.



Good First Contributions
========================

If you would like to contribute to Qiskit, but aren't sure
where to get started, the ``good first issue`` label on issues for a project
highlights items appropriate for people new to the project.
These are all issues that have been reviewed and tagged by contributors
as something a new contributor should be able to work on. In other
words, intimate familiarity with Qiskit is not a requirement to develop a fix
for the issue.



Deprecation Policy
==================

Qiskit users need to know if a feature or an API they rely
upon will continue to be supported by the software tomorrow. Knowing under which conditions
the project can remove (or change in a backwards-incompatible manner) a feature or
API is important to the user. To manage expectations, the following policy is how API
and feature deprecation/removal is handled by Qiskit:

1. Features, APIs, or configuration options are marked deprecated in the code.
Appropriate ``DeprecationWarning`` class warnings will be sent to the user. The
deprecated code will be frozen and receive only minimal maintenance (just so
that it continues to work as-is).

2. A migration path will be documented for current users of the feature. This
will be outlined in the both the release notes adding the deprecation, and the
release notes removing the feature at the completion of the deprecation cycle.
If feasible, the warning message will also include the migration
path. A migration path might be "stop using that feature", but in such cases
it is necessary to first judge how widely used and/or important the feature
is to users, in order to determine a reasonable obsolescence date.

3. An obsolescence date for the feature will be set. The feature must remain
intact and working (although with the proper warning being emitted) in all
releases pushed until after that obsolescence date. At the very minimum, the
feature (or API, or configuration option) should be marked as deprecated (and
continue to be supported) for at least three months of linear time from the release
date of the first release to include the deprecation warning. For example, if a
feature were deprecated in the 0.9.0 release of Terra, which was released on
August 22, 2019, then that feature should still appear in all releases until at
least November 22, 2019. Since releases do not occur at fixed time intervals,
a deprecation warning may only occur in one release prior to removal.

Note that this delay is a minimum. For significant features, it is recommended
that the deprecated feature appears for at least double that time. Also, per
the stable branch policy, deprecation removals can only occur during minor
version releases; they are not appropriate for backporting.



Deprecation Warnings
--------------------

The proper way to raise a deprecation warning is to use the ``warn`` function
from the `warnings module <https://docs.python.org/3/library/warnings.html>`__
in the Python standard library. The warning category class
should be a ``DeprecationWarning``. An example would be::

 import warnings

 def foo(input):
     warnings.warn('The qiskit.foo() function is deprecated as of 0.9.0, and '
                   'will be removed no earlier than 3 months after that '
                   'release date. You should use the qiskit.bar() function '
                   'instead.', DeprecationWarning, stacklevel=2)

One thing to note here is the ``stack_level`` kwarg on the warn() call. This
argument is used to specify which level in the call stack will be used as
the line initiating the warning. Typically, ``stack_level`` should be set to 2,
as this will show the line calling the context where the warning was raised.
In the above example, it would be the caller of ``foo()``. If you did not set this,
the warning would show that it was caused by the line in the foo()
function, which is not helpful for users trying to determine the origin
of a deprecated call. However, this value may be adjusted, depending on the call
stack and where ``warn()`` gets called from. For example, if the warning is always
raised by a private method that only has one caller, ``stack_level=3`` might be
appropriate.



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



*****************************
Contributing to Documentation
*****************************

Qiskit documentation is shaped by the `docs as code
<https://www.writethedocs.org/guide/docs-as-code/>`__ philosophy, primarily
drawn from Qiskit code comments in the `style of API documentation
<https://alistapart.com/article/the-ten-essentials-for-good-api-documentation/>`__.

The documentation is built from the master branch of `Qiskit/qiskit/docs
<https://github.com/Qiskit/qiskit/tree/master/docs>`__ using `Sphinx
<http://www.sphinx-doc.org/en/master/>`__. The majority of documentation, under
`API Reference <https://qiskit.org/documentation/apidoc/qiskit.html>`__, is
drawn from code comments in the repositories listed in :ref:`where_things_are`.



Documentation Structure
=======================

The way documentation is structured in Qiskit is to push as much of the actual
documentation into the docstrings as possible. This makes it easier for
additions and corrections to be made during development, because the majority
of the documentation lives near the code being changed. There are three levels in
the normal documentation structure in Terra:

The ``.rst`` files in the ``docs/apidocs``
   These files are used to tell Sphinx which modules to include in the rendered
   documentation. This contains two pieces of information:
   an `internal reference <http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#reference-names>`__
   or `cross reference <https://www.sphinx-doc.org/en/latest/usage/restructuredtext/roles.html#ref-role>`__
   to the module, which can be used for internal links
   inside the documentation, and an `automodule directive <http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`__
   used to parse the
   module docstrings from a specified import path. For example, the dagcircuit.rst
   file contains::

      .. _qiskit-dagcircuit:


      .. automodule:: qiskit.dagcircuit
         :no-members:
         :no-inherited-members:
         :no-special-members:

   The only ``.rst`` file outside of this is ``qiskit.rst``, which contains the table of
   contents. If you're adding a new ``.rst`` file for a new module's documentation, make
   sure to add it to the `toctree <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-of-contents>`__
   in that file.

The module-level docstring
   This docstring is at the module
   level for the module specified in the ``automodule`` directive in the rst file.
   If the module specified is a directory/namespace, the docstring should be
   specified in the ``__init__.py`` file for that directory. This module-level
   docstring contains more details about the module being documented.
   The normal structure to this docstring is to outline all the classes and
   functions of the public API that are contained in that module. This is typically
   done using the `autosummary directive <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`__
   (or `autodoc directives <http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`__
   directly if the module is simple, such as in the case of ``qiskit.execute``). The
   autosummary directive is used to autodoc a list of different Python elements
   (classes, functions, etc.) directly without having to manually call out the
   autodoc directives for each one. The module-level docstring is where to
   provide a high-level overview of what functionality the module provides.
   This is normally done by grouping the different
   components of the public API together into multiple subsections.

   For example, as in the previous dagcircuit module example, the
   contents of the module docstring for ``qiskit/dagcircuit/__init__.py`` would
   be::

      """
      =======================================
      DAG Circuits (:mod:`qiskit.dagcircuit`)
      =======================================
      .. currentmodule:: qiskit.dagcircuit
      DAG Circuits
      ============
      .. autosummary::
         :toctree: ../stubs/
         DAGCircuit
         DAGNode
      Exceptions
      ==========
      .. autosummary::
         :toctree: ../stubs/
         DAGCircuitError
      """

   .. note::

      This is just an example and the actual module docstring for the dagcircuit
      module might diverge from this.

The actual docstring for the elements listed in the module docstring
   You should strive to document thoroughly all the public interfaces
   exposed using examples when necessary. For docstrings, `Google Python Style
   Docstrings <https://google.github.io/styleguide/pyguide.html?showone=Comments#38-comments-and-docstrings>`__
   are used. This is parsed using the `napoleon
   sphinx extension <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`__.
   The `napoleon documentation <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__
   contains a good example of how docstrings should be formatted.

   .. note::
      You can use any Sphinx directive or rst formatting in a docstring as it
      makes sense. For example, one common extension used is the ``jupyter-execute``
      directive, which is used to execute a code block in Jupyter and display both
      the code and output. This is particularly useful for visualizations.



Documentation Integration
-------------------------

The hosted documentation at https://qiskit.org/documentation/ covers the entire
Qiskit project; Terra is just one component of that. As such, the documentation
builds for the hosted version are built by the Qiskit meta-package repository
https://github.com/Qiskit/qiskit. When commits are merged to that repo, the
output of Sphinx builds are uploaded to the qiskit.org website. Those Sphinx
builds are configured to pull in the documentation from the version of the
Qiskit elements installed by the meta-package at that point. For example, if
the meta-package version is currently 0.13.0, then that will copy the
documentation from Terra's 0.10.0 release. When the meta-package's requirements
are bumped, then it will start pulling documentation from the new version. This
means that fixes for incorrect API documentation will need to be
included in a new release. Documentation fixes are valid backports for a stable
patch release per the stable branch policy (see that section below).

During the build process, the contents of each element's ``docs/apidocs/``
are recursively copied into a shared copy of ``doc/apidocs/`` in the meta-package
repository along with all the other elements. This means that what is in the root of
docs/apidocs on each element at a release will end up on the root of
https://qiskit.org/documentation/apidoc/.



Translating Documentation
=========================

Qiskit documentation is translated (localized) using Crowdin, a software and web
localization platform that allows organizations to coordinate translation
projects and collaborate with communities to translate materials. Crowdin allows
our community of translators to amplify their impact by automatically reusing
the work invested translating one sentence to translate other, similar
sentences. Crowdin also makes translations resilient to many types of changes to
the original material, such as moving sentences around, even across files.

Qiskit localization requests are handled in `Qiskit Translations <https://github.com/Qiskit/qiskit-translations>`__
repository. To contribute to Qiskit localization, please follow these steps:

#. Add your name (or ID) to the `LOCALIZATION_CONTRIBUTORS
   <https://github.com/qiskit-community/qiskit-translations/blob/master/LOCALIZATION_CONTRIBUTORS>`__
   file.
#. Create a pull request (PR) to merge your change. Make sure to follow the template
   to open a Pull Request.

   .. note::

      - Each contributor has to create their own PR and sign the CLA.
      - Please mention the language that you'd like to contribute to in the PR
        summary.
      - If you have an open issue for a language request, **add the issue link
        to the PR**.
#. You will be asked to sign the Qiskit Contributors License Agreement (CLA);
   please do so.
#. A minimum of **three contributors** per language are necessary for any new
   languages to be added, to receive official support from the administrators of
   the localization project.
#. Among the group of contributors, a translation lead must be identified to serve
   as a liaison with the administrators of the localization project.
   The lead must contact: Yuri Kobayashi (yurik@jp.ibm.com) by email.
#. In the `Qiskit-Docs <https://crowdin.com/project/qiskit-docs>`__
   Crowdin project, choose the language that you want to contribute to.

   .. note::

      As mentioned in the blog post, `Qiskit in my language is Qiskit <https://medium.com/qiskit/qiskit-in-my-language-is-qiskit-73d4626a99d3>`__,
      we want to make sure that translated languages have enough community support
      to build a translation team with translators, proofreaders, and translation leads.
      If you want to be a translation lead or would be willing to join a new
      translation project team, you can open a `GitHub issue <https://github.com/qiskit-community/qiskit-translations/issues/new/choose>`__
      to start a discussion with the Qiskit team and recruit translation project members.
#. Click the **Join** button and **paste the URL of your PR** in the dialog box where you
   are asked why you want to join the Crowdin project.

The administrators of the Crowdin project will review your request and give you
access as quickly as they can.

Building from Source
====================

You can build a local copy of the documentation from your local clone of the
`Qiskit/qiskit` repository as follows:

1. Clone the Qiskit repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit.git

2. Cloning the repository creates a local folder called ``qiskit``.

   .. code:: sh

      cd qiskit

3. Build the documentation by navigating to your local clone of `Qiskit/qiskit`
   and running the following command in a terminal window.

   .. code-block:: sh

      tox -edocs

   If you do not already have the `tox <https://tox.readthedocs.io/en/latest/>`_
   command installed, install it by running:

   .. code:: sh

      pip install tox

As you make changes to your local RST files, you can update your
HTML files by navigating to `/doc/` and running the following in a terminal
window:

   .. code-block:: sh

      tox -edocs

This will build a styled, HTML version of your local documentation repository
in the subdirectory `/docs/_build/html/`.

.. _install_install_from_source_label:

**********************
Installing from Source
**********************

Installing the elements from source allows you to access the most recently
updated version of Qiskit instead of using the version in the Python Package
Index (PyPI) repository. This will give you the ability to inspect and extend
the latest version of the Qiskit code more efficiently.

When installing the elements and components from source, by default their
``development`` version (which corresponds to the ``master`` git branch) will
be used, as opposed to the ``stable`` version (which contains the same codebase
as the published ``pip`` packages). Since the ``development`` versions of an
element or component usually include new features and changes, they generally
require using the ``development`` version of the rest of the items as well.

.. note::

  The Terra and Aer packages both require a compiler to build from source before
  you can install. Ignis, Aqua, and the IBM Quantum Provider backend
  do not require a compiler.

Installing elements from source requires the following order of installation to
prevent installing versions of elements that may be lower than those desired if the
``pip`` version is behind the source versions:

#. :ref:`qiskit-terra <install-qiskit-terra>`
#. :ref:`qiskit-aer <install-qiskit-aer>`
#. :ref:`qiskit-ignis <install-qiskit-ignis>`
#. :ref:`qiskit-aqua <install-qiskit-aqua>`
#. :ref:`qiskit-ibmq-provider <install-qiskit-ibmq-provider>`
   (if you want to connect to the IBM Quantum devices or online
   simulator)

To work with several components and elements simultaneously, use the following
steps for each element.

.. note::

   Due to the use of namespace packaging in Python, care must be taken in how you
   install packages. If you're planning to install any element from source, do not
   use the ``qiskit`` meta-package. Also, follow this guide and use a separate virtual
   environment for development. If you do choose to mix an existing installation
   with your development, refer to
   https://github.com/pypa/sample-namespace-packages/blob/master/table.md
   for the set of combinations of installation methods that work together.

Set up the Virtual Development Environment
==========================================

.. code-block:: sh

   conda create -y -n QiskitDevenv python=3
   conda activate QiskitDevenv

.. _install-qiskit-terra:

Installing Terra from Source
============================

Installing from source requires that you have a C++ compiler on your system that supports
C++11.


.. tabbed:: Compiler for Linux

   On most Linux platforms, the necessary GCC compiler is already installed.

.. tabbed:: Compiler for macOS

   If you use macOS, you can install the Clang compiler by installing XCode.
   Check if you have XCode and Clang installed by opening a terminal window and entering the
   following.

   .. code:: sh

      clang --version

   Install XCode and Clang by using the following command.

   .. code:: sh

      xcode-select --install

.. tabbed:: Compiler for Windows

   On Windows, it is easiest to install the Visual C++ compiler from the
   `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`__.
   You can instead install Visual Studio version 2015 or 2017, making sure to select the
   options for installing the C++ compiler.


Once the compilers are installed, you are ready to install Qiskit Terra.

1. Clone the Terra repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-terra.git

2. Cloning the repository creates a local folder called ``qiskit-terra``.

   .. code:: sh

      cd qiskit-terra

3. Install the Python requirements libraries from your ``qiskit-terra`` directory.

   .. code:: sh

      pip install cython

4. If you want to run tests or linting checks, install the developer requirements.

   .. code:: sh

      pip install -r requirements-dev.txt

5. Install ``qiskit-terra``.

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied, you can do this with:

.. code:: sh

   pip install -e .

You can then run the code examples after installing Terra. You can
run the example with the following command.

.. code:: sh

   python examples/python/using_qiskit_terra_level_0.py


.. note::

    If you do not intend to install any other components, qiskit-terra will
    emit a ``RuntimeWarning`` warning that both qiskit-aer and
    qiskit-ibmq-provider are not installed. This is done because
    users commonly intend to use the additional elements,
    but do not realize they are not installed, or that the installation
    of either Aer or the IBM Quantum Provider failed for some reason. If you wish
    to suppress these warnings, add::

        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                module='qiskit')

    before any ``qiskit`` imports in your code. This will suppress the
    warning about the missing qiskit-aer and qiskit-ibmq-provider, but
    will continue to display any other warnings from qiskit or other packages.

.. _install-qiskit-aer:

Installing Aer from Source
==========================

1. Clone the Aer repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-aer

2. Install build requirements.

   .. code:: sh

      pip install cmake scikit-build cython

After this, the steps to install Aer depend on which operating system you are
using. Since Aer is a compiled C++ program with a Python interface, there are
non-Python dependencies for building the Aer binary which can't be installed
universally depending on operating system.

.. tabbed:: Linux

   3. Install compiler requirements.

      Building Aer requires a C++ compiler and development headers.

      If you're using Fedora or an equivalent Linux distribution,
      install using:

      .. code:: sh

         dnf install @development-tools

      For Ubuntu/Debian install it using:

      .. code:: sh

         apt-get install build-essential

   4. Install OpenBLAS development headers.

      If you're using Fedora or an equivalent Linux distribution,
      install using:

      .. code:: sh

         dnf install openblas-devel

      For Ubuntu/Debian install it using:

      .. code:: sh

         apt-get install libopenblas-dev


.. tabbed:: macOS

   3. Install dependencies.

      To use the `Clang <https://clang.llvm.org/>`__ compiler on macOS, you need to install
      an extra library for supporting `OpenMP <https://www.openmp.org/>`__.  You can use `brew <https://brew.sh/>`__
      to install this and other dependencies.

      .. code:: sh

         brew install libomp

   4. Then install a BLAS implementation; `OpenBLAS <https://www.openblas.net/>`__
      is the default choice.

      .. code:: sh

         brew install openblas

      Next, install ``Xcode Command Line Tools``.

      .. code:: sh

         xcode-select --install

.. tabbed:: Windows

   On Windows you need to use `Anaconda3 <https://www.anaconda.com/distribution/#windows>`__
   or `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`__ to install all the
   dependencies.

   3. Install compiler requirements.

      .. code:: sh

         conda install --update-deps vs2017_win-64 vs2017_win-32 msvc_runtime

   4. Install binary and build dependencies.

      .. code:: sh

         conda install --update-deps -c conda-forge -y openblas cmake


5. Build and install qiskit-aer directly

   If you have pip <19.0.0 installed and your environment doesn't require a
   custom build, run:

   .. code:: sh

      cd qiskit-aer
      pip install .

   This will both build the binaries and install Aer.

   Alternatively, if you have a newer pip installed, or have some custom requirement,
   you can build a Python wheel manually.

   .. code:: sh

      cd qiskit-aer
      python ./setup.py bdist_wheel

   If you need to set a custom option during the wheel build, refer to
   :ref:`aer_wheel_build_options`.

   After you build the Python wheel, it will be stored in the ``dist/`` dir in the
   Aer repository. The exact version will depend

   .. code:: sh

      cd dist
      pip install qiskit_aer-*.whl

   The exact filename of the output wheel file depends on the current version of
   Aer under development.

.. _aer_wheel_build_options:

Custom options during wheel builds
----------------------------------

The Aer build system uses `scikit-build <https://scikit-build.readthedocs.io/en/latest/index.html>`__
to run the compilation when building it with the Python interface. It acts as an interface for
`setuptools <https://setuptools.readthedocs.io/en/latest/>`__ to call `CMake <https://cmake.org/>`__
and compile the binaries for your local system.

Due to the complexity of compiling the binaries, you may need to pass options
to a certain part of the build process. The way to pass variables is:

.. code:: sh

   python setup.py bdist_wheel [skbuild_opts] [-- [cmake_opts] [-- build_tool_opts]]

where the elements within square brackets `[]` are optional, and
``skbuild_opts``, ``cmake_opts``, ``build_tool_opts`` are to be replaced by
flags of your choice. A list of *CMake* options is available here:
https://cmake.org/cmake/help/v3.6/manual/cmake.1.html#options. For
example, you could run something like:

.. code:: sh

   python setup.py bdist_wheel -- -- -j8

This is passing the flag `-j8` to the underlying build system (which in this
case is `Automake <https://www.gnu.org/software/automake/>`__), telling it that you want
to build in parallel using 8 processes.

For example, a common use case for these flags on linux is to specify a
specific version of the C++ compiler to use (normally if the default is too
old):

.. code:: sh

   python setup.py bdist_wheel -- -DCMAKE_CXX_COMPILER=g++-7

which will tell CMake to use the g++-7 command instead of the default g++ when
compiling Aer.

Another common use case for this, depending on your environment, is that you may
need to specify your platform name and turn off static linking.

.. code:: sh

   python setup.py bdist_wheel --plat-name macosx-10.9-x86_64 \
   -- -DSTATIC_LINKING=False -- -j8

Here ``--plat-name`` is a flag to setuptools, to specify the platform name to
use in the package metadata, ``-DSTATIC_LINKING`` is a flag for using CMake
to disable static linking, and ``-j8`` is a flag for using Automake to use
8 processes for compilation.

A list of common options depending on platform are:

+--------+------------+----------------------+---------------------------------------------+
|Platform| Tool       | Option               | Use Case                                    |
+========+============+======================+=============================================+
| All    | Automake   | -j                   | Followed by a number, sets the number of    |
|        |            |                      | processes to use for compilation.           |
+--------+------------+----------------------+---------------------------------------------+
| Linux  | CMake      | -DCMAKE_CXX_COMPILER | Used to specify a specific C++ compiler;    |
|        |            |                      | this is often needed if your default g++ is |
|        |            |                      | too old.                                    |
+--------+------------+----------------------+---------------------------------------------+
| OSX    | setuptools | --plat-name          | Used to specify the platform name in the    |
|        |            |                      | output Python package.                      |
+--------+------------+----------------------+---------------------------------------------+
| OSX    | CMake      | -DSTATIC_LINKING     | Used to specify whether or not              |
|        |            |                      | static linking should be used.              |
+--------+------------+----------------------+---------------------------------------------+

.. note::
    Some of these options are not platform-specific. These particular platforms are listed
    because they are commonly used in the environment. Refer to the
    tool documentation for more information.

.. _install-qiskit-ignis:

Installing Ignis from Source
============================

1. Clone the Ignis repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-ignis.git

2. Cloning the repository creates a local directory called ``qiskit-ignis``.

   .. code:: sh

      cd qiskit-ignis

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-ignis package when installing
   from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install Ignis.

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied:

.. code:: sh

    pip install -e .

.. _install-qiskit-aqua:

Installing Aqua from Source
===========================

1. Clone the Aqua repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-aqua.git

2. Cloning the repository creates a local directory called ``qiskit-aqua``.

   .. code:: sh

      cd qiskit-aqua

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-aqua package when installing
   from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install Aqua.

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied:

.. code:: sh

    pip install -e .

.. _install-qiskit-ibmq-provider:

Installing IBM Quantum Provider from Source
===========================================

1. Clone the qiskit-ibmq-provider repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-ibmq-provider.git

2. Cloning the repository creates a local directory called ``qiskit-ibmq-provider``.

   .. code:: sh

      cd qiskit-ibmq-provider

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-ibmq-provider package when
   installing from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install qiskit-ibmq-provider.

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied:

.. code:: sh

    pip install -e .
