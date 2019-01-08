
Contributing
============

**We appreciate all kinds of help, so thank you!**


Contributing to Qiskit Terra
----------------------------

You can contribute in many ways to this project.


Issue reporting
~~~~~~~~~~~~~~~

This is a good point to start, when you find a problem please add
it to the `issue tracker <https://github.com/Qiskit/qiskit-terra/issues>`_.
The ideal report should include the steps to reproduce it.


Doubts solving
~~~~~~~~~~~~~~

To help less advanced users is another wonderful way to start. You can
help us close some opened issues. This kind of tickets should be
labeled as ``question``.


Improvement proposal
~~~~~~~~~~~~~~~~~~~~

If you have an idea for a new feature please open a ticket labeled as
``enhancement``. If you could also add a piece of code with the idea
or a partial implementation it would be awesome.


Contributor License Agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'd love to accept your code! Before we can, we have to get a few legal
requirements sorted out. By signing a contributor license agreement (CLA), we
ensure that the community is free to use your contributions.

When you contribute to the Qiskit Terra project with a new pull request, a bot will
evaluate whether you have signed the CLA. If required, the bot will comment on
the pull request,  including a link to accept the agreement. The
`individual CLA <https://qiskit.org/license/qiskit-cla.pdf>`_ document is
available for review as a PDF.

.. note::
    If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a `corporate CLA <https://qiskit.org/license/qiskit-corporate-cla.pdf>`_
    and email it to us at qiskit@us.ibm.com.


Good first contributions
~~~~~~~~~~~~~~~~~~~~~~~~

You are welcome to contribute wherever in the code you want to, of course, but
we recommend taking a look at the "Good first contribution" label into the
issues and pick one. We would love to mentor you!


Doc
~~~

Review the parts of the documentation regarding the new changes and update it
if it's needed.


Pull requests
~~~~~~~~~~~~~

We use `GitHub pull requests <https://help.github.com/articles/about-pull-requests>`_
to accept the contributions.

A friendly reminder! We'd love to have a previous discussion about the best way to
implement the feature/bug you are contributing with. This is a good way to
improve code quality in our beloved Qiskit!, so remember to file a new Issue before
starting to code for a solution.

So after having discussed the best way to land your changes into the codebase,
you are ready to start coding (yay!). We have two options here:

1. You think your implementation doesn't introduce a lot of code, right?. Ok,
   no problem, you are all set to create the PR once you have finished coding.
   We are waiting for it!
2. Your implementation does introduce many things in the codebase. That sounds
   great! Thanks!. In this case you can start coding and create a PR with the
   word: **[WIP]** as a prefix of the description. This means "Work In
   Progress", and allow reviewers to make micro reviews from time to time
   without waiting for the big and final solution... otherwise, it would make
   reviewing and coming changes pretty difficult to accomplish. The reviewer
   will remove the **[WIP]** prefix from the description once the PR is ready
   to merge.


Pull request checklist
""""""""""""""""""""""

When submitting a pull request and you feel it is ready for review, please
double check that:

* the code follows the code style of the project. For convenience, you can
  execute ``make style`` and ``make lint`` locally, which will print potential
  style warnings and fixes.
* the documentation has been updated accordingly. In particular, if a function
  or class has been modified during the PR, please update the docstring
  accordingly.
* your contribution passes the existing tests, and if developing a new feature,
  that you have added new tests that cover those changes.
* you add a new line to the ``CHANGELOG.rst`` file, in the ``UNRELEASED``
  section, with the title of your pull request and its identifier (for example,
  "``Replace OldComponent with FluxCapacitor (#123)``".


Commit messages
"""""""""""""""

Please follow the next rules for the commit messages:

- It should include a reference to the issue ID in the first line of the commit,
  **and** a brief description of the issue, so everybody knows what this ID
  actually refers to without wasting to much time on following the link to the
  issue.

- It should provide enough information for a reviewer to understand the changes
  and their relation to the rest of the code.

A good example:

.. code-block:: text

    Issue #190: Short summary of the issue
    * One of the important changes
    * Another important change


Installing Qiskit Terra from source
-----------------------------------

This section include some tips that will help you install and push source code.

.. note::

    We recommend using `Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`__
    to cleanly separate Qiskit from other applications and improve your experience.


Setup with an environment
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use environments is by using Anaconda

.. code:: sh

    conda create -y -n QiskitDevenv python=3
    source activate QiskitDevenv

For the python code, we need some libraries that can be installed in this way:

.. code:: sh

    cd qiskit-terra
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

To get the examples working try  

.. code:: sh

    $ pip install -e .
 
and then you can run them with 

.. code:: sh

    $ python examples/python/using_qiskit_terra_level_0.py

We recommend that after setting up Terra you set up Aer to get more advanced simulators.  


Test
~~~~

New features often imply changes in the existent tests or new ones are
needed. Once they're updated/added run this be sure they keep passing.

For executing the tests, a ``make test`` target is available.
The execution of the tests (both via the make target and during manual invocation)
takes into account the ``LOG_LEVEL`` environment variable. If present, a ``.log``
file will be created on the test directory with the output of the log calls, which
will also be printed to stdout. You can adjust the verbosity via the content
of that variable, for example:

Linux and Mac:

.. code-block:: bash

    $ cd out
    out$ LOG_LEVEL="DEBUG" ARGS="-V" make test

Windows:

.. code-block:: bash

    $ cd out
    C:\..\out> set LOG_LEVEL="DEBUG"
    C:\..\out> set ARGS="-V"
    C:\..\out> make test

For executing a simple python test manually, we don't need to change the directory
to ``out``, just run this command:


Linux and Mac:

.. code-block:: bash

    $ LOG_LEVEL=INFO python -m unittest test/python/circuit/test_circuit_operations.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/circuit/test_circuit_operations.py

Note many of the tests will not be executed unless you have setup an IBMQ
account. To set this up please go to this
`page <https://quantumexperience.ng.bluemix.net/qx/account/advanced>`_  and
register an account.

By default, and if there is no user credentials available, the tests that
require online access are run with recorded (mocked) information. This is, the
remote requests are replayed from a ``test/cassettes`` and not real HTTP
requests is generated. If user credentials are found, in that cases it use them
to make the network requests.

How and which tests are executed is controlled by a environment variable
``QISKIT_TESTS``. The options are (where ``uc_available = True`` if the user
credentials are available, and ``False`` otherwise):

+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
|  Option           | Description                                                                                                        | Default               |  If ``True``, forces                             |
+===================+====================================================================================================================+=======================+==================================================+
| ``skip_online``   | Skips tests that require remote requests (also, no mocked information is used). Does not require user credentials. | ``False``             | ``rec = False``                                  |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``mock_online``   | It runs the online tests using mocked information. Does not require user credentials.                              | ``not uc_available``  | ``skip_online = False``                          |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``run_slow``      | It runs tests tagged as *slow*.                                                                                    | ``False``             |                                                  |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+
| ``rec``           | It records the remote requests. It requires user credentials.                                                      | ``False``             | ``skip_online = False``                          |
|                   |                                                                                                                    |                       | ``run_slow = False``                             |
+-------------------+--------------------------------------------------------------------------------------------------------------------+-----------------------+--------------------------------------------------+

It is possible to provide more than one option separated with commas.
The order of precedence in the options is right to left. For example,
``QISKIT_TESTS=skip_online,rec`` will set the options as
``skip_online == False`` and ``rec == True``.

Style guide
~~~~~~~~~~~

Please submit clean code and please make effort to follow existing conventions
in order to keep it as readable as possible. We use
`Pylint <https://www.pylint.org>`_ and `PEP
8 <https://www.python.org/dev/peps/pep-0008>`_ style guide: to ensure
your changes respect the style guidelines, run the next commands:

All platforms:

.. code:: sh

    $> cd out
    out$> make lint
    out$> make style


Documentation
-------------

The documentation Qiskit Terra is in the ``doc`` directory. The
documentation is auto-generated from python
docstrings using `Sphinx <http://www.sphinx-doc.org>`_ for generating the
documentation. Please follow `Google's Python Style
Guide <https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_
for docstrings. A good example of the style can also be found with
`sphinx's napolean converter
documentation <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
You can see the rendered documentation for the stable version of Qiskit Terra at
the `landing page <https://qiskit.org/terra>`_.

All platforms:

.. code:: sh

    $> make doc


Development cycle
-----------------

Our development cycle is straightforward, use the project boards in Github 
for project management and use milestones for releases. The features 
that we want to include in these releases will be tagged and discussed 
in the project boards. Whenever a new release is close to be launched, 
we'll announce it and detail what has changed since the latest version in 
our Release Notes and Changelog. The channels we'll use to announce new 
releases are still being discussed, but for now, you can 
`follow us <https://twitter.com/qiskit>`_ on Twitter!


Branch model
~~~~~~~~~~~~

There are two main branches in the repository:

- ``master``

  - This is the development branch.
  - Next release is going to be developed here. For example, if the current
    latest release version is r1.0.3, the master branch version will point to
    r1.1.0 (or r2.0.0).
  - You should expect this branch to be updated very frequently.
  - Even though we are always doing our best to not push code that breaks
    things, is more likely to eventually push code that breaks something...
    we will fix it ASAP, promise :).
  - This should not be considered as a stable branch to use in production
    environments.
  - The API of Qiskit could change without prior notice.

- ``stable``

  - This is our stable release branch.
  - It's always synchronized with the latest distributed package, as for now,
    the package you can download from pip.
  - The code in this branch is well tested and should be free of errors
    (unfortunately sometimes it's not).
  - This is a stable branch (as the name suggest), meaning that you can expect
    stable software ready for production environments.
  - All the tags from the release versions are created from this branch.


Release cycle
~~~~~~~~~~~~~

From time to time, we will release brand new versions of Qiskit Terra. These
are well-tested versions of the software.

When the time for a new release has come, we will:

1. Merge the ``master`` branch with the ``stable`` branch.
2. Create a new tag with the version number in the ``stable`` branch.
3. Crate and distribute the pip package.
4. Change the ``master`` version to the next release version.
5. Announce the new version to the world!

The ``stable`` branch should only receive changes in the form of bug fixes, so the
third version number (the maintenance number: [major].[minor].[maintenance])
will increase on every new change.

