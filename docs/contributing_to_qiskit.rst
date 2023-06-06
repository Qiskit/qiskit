######################
Contributing to Qiskit
######################

Qiskit is an open-source project committed to bringing quantum computing to
people of all backgrounds. This page describes how you can join the Qiskit
community in this goal.

****************
Before You Start
****************

If you are new to Qiskit contributing we recommend you do the following before diving into the code:

#. Read the `Code of Conduct <https://github.com/Qiskit/qiskit/blob/master/CODE_OF_CONDUCT.md>`__
#. :ref:`Decide what to work on <decide-what-to-work-on>`
#. Read the repo-specific :ref:`Contributing Guidelines <contributing_links>` for the repo you have decided to contribute to.
#. :ref:`Set up your development environment <dev-env-setup>`
#. Familiarize yourself with the Qiskit community (via `Slack <https://qisk.it/join-slack>`__,
   `Stack Exchange <https://quantumcomputing.stackexchange.com/>`__, `GitHub <https://github.com/qiskit-community/feedback/discussions>`__ etc.)


.. _decide-what-to-work-on:

************************
Decide What to Work on
************************

If you're not sure what type of contribution is right for you, take a look at the following flowchart to help you:

.. raw:: html
   :file: images/contributor-flowchart.svg

.. _contributing_links:

********************************
Contributing to a Specific Repo
********************************

Each Qiskit package has its own set of Contributing Guidelines (kept in the ``CONTRIBUTING.md`` file) which
details specific information on contributing to that repository. Make sure you read through the repo-specific
Contributing Guidelines prior to making your contribution to a specific repo as each project may have
slightly different requirements and processes. For Qiskit Terra, the main repository, the contributing guidelines
may be be found `here <https://github.com/Qiskit/qiskit-terra/blob/main/CONTRIBUTING.md>`__. Other Qiskit packages that
are able to receive contributions may be found as seperate repositories in the official `Qiskit Github <https://github.com/Qiskit>`__.

.. _dev-env-setup:

Set up Your Development Environment
===================================

To get started contributing to the Python-based Qiskit repos you will need to set up a Python Virtual
Development Environment and install the appropriate package **from source**.

For a quick guide on how to do this for qiskit-terra take a look at the
`How to Install Qiskit - Contributors <https://www.youtube.com/watch?v=Pix2MFCtiOo>`__ YouTube video.

For non-python packages you should check the CONTRIBUTING.md file for specific details on setting up your dev environment.

Set up Python Virtual Development Environment
---------------------------------------------

Virtual environments are used for Qiskit development to isolate the development environment
from system-wide packages. This way, we avoid inadvertently becoming dependent on a
particular system configuration. For developers, this also makes it easy to maintain multiple
environments (e.g. one per supported Python version, for older versions of Qiskit, etc.).

.. tab-set::

    .. tab-item:: Python venv

       All Python versions supported by Qiskit include built-in virtual environment module
       `venv <https://docs.python.org/3/tutorial/venv.html>`__.

       Start by creating a new virtual environment with ``venv``. The resulting
       environment will use the same version of Python that created it and will not inherit installed
       system-wide packages by default. The specified folder will be created and is used to hold the environment's
       installation. It can be placed anywhere. For more detail, see the official Python documentation,
       `Creation of virtual environments <https://docs.python.org/3/library/venv.html>`__.

       .. code-block:: sh

          python3 -m venv ~/.venvs/qiskit-dev

       Activate the environment by invoking the appropriate activation script for your system, which can
       be found within the environment folder. For example, for bash/zsh:

       .. code-block:: sh

          source ~/.venvs/qiskit-dev/bin/activate

       Upgrade pip within the environment to ensure Qiskit dependencies installed in the subsequent sections
       can be located for your system.

       .. code-block:: sh

          pip install -U pip

    .. tab-item:: Conda

       For Conda users, a new environment can be created as follows.

       .. code-block:: sh

          conda create -y -n QiskitDevenv python=3
          conda activate QiskitDevenv



.. code:: sh

    pip install -e .


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
to finishing the implementation, you can open a *Draft* pull request and prepend
the title with the **\[WIP\]** tag (for Work In Progress). This will indicate
to reviewers that the code in the PR isn't in its final state and will change.
It also means that we will not merge the commit until it is finished. You or a
reviewer can remove the [WIP] tag when the code is ready to be fully reviewed for merging.

Before marking your Pull Request as "ready for review" make sure you have followed the
PR Checklist below. PRs that adhere to this list are more likely to get reviewed and
merged in a timely manner.

.. _pr-checklist:

**Pull Request Checklist:**
---------------------------
- You have followed the requirements in the CONTRIBUTING.md file for the specific repo you are
  contributing to.
- All CI checks pass (it's recommended to run tests and lint checks locally before pushing).
- New tests have been added for any new functionality that has been introduced.
- The documentation has been updated accordingly for any new/modified functionality.
- A release note has been added if the change has a user-facing impact.
- Any superfluous comments or print statements have been removed.
- All contributors have signed the :ref:`cla`.
- The PR has a concise and explanatory title (e.g. ``Fixes Issue1234`` is a bad title!).
- If the PR addresses an open issue the PR description includes the ``fixes #issue-number``
  syntax to link the PR to that issue (**you must use the exact phrasing in order for GitHub
  to automatically close the issue when the PR merges**)



Code Review
===========

Code review is done in the open and is open to anyone. While only maintainers have
access to merge commits, community feedback on pull requests is extremely valuable.
It is also a good mechanism to learn about the code base.

Response times may vary for your PR, it is not unusual to wait a few weeks for a maintainer
to review your work, due to other internal commitments. If you have been waiting over a week
for a review on your PR feel free to tag the relevant maintainer in a comment to politely remind
them to review your work.

Please be patient! Maintainers have a number of other priorities to focus on and so it may take
some time for your work to get reviewed and merged. PRs that are in a good shape (i.e. following the :ref:`pr-checklist`)
are easier for maintainers to review and more likely to get merged in a timely manner. Please also make
sure to always be kind and respectful in your interactions with maintainers and other contributors, you can read
`the Qiskit Code of Conduct <https://github.com/Qiskit/qiskit/blob/master/CODE_OF_CONDUCT.md>`__.



.. _cla:

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
