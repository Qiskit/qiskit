Contributing
============

**We appreciate all kinds of help, so thank you!**

Contributing to the project
---------------------------

You can contribute in many ways to this project.

Issue reporting
~~~~~~~~~~~~~~~

This is a good point to start, when you find a problem please add
it to the `issue tracker <https://github.com/QISKit/qiskit-sdk-py/issues>`_.
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

Code
----

This section include some tips that will help you to push source code.

Dependencies
~~~~~~~~~~~~

Our build system is based on CMake, so we need to have `CMake 3.5 or higher <https://cmake.org/>`_
installed. As we will deal with languages that build native binaries, we will
need to have installed any of the `supported CMake build tools <https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html>`_.

On Linux and Mac, we recommend installing GNU g++ 6.1 or higher, on Windows
we only support `MinGW64 <http://mingw-w64.org>`_ at the moment.
Note that a prerequiste for the C++ toolchain is that C++14 must be supported.

For the python code, we need some libraries that can be installed in this way:

.. code:: sh

    # Depending on the system and setup to append "sudo -H" before could be needed.
    pip install -U -r requirements.txt
    pip install -U -r requirements-dev.txt

Building
~~~~~~~~

The preferred way CMake is meant to be used, is by setting up an "out of source" build.
So in order to build our native code, we have to follow these steps:

Linux and Mac

.. code::

    qiskit-sdk-py$ mkdir out
    qiskit-sdk-py$ cd out
    qiskit-sdk-py/out$ cmake ..
    qiskit-sdk-py/out$ make

Windows

.. code::

    C:\..\> mkdir out
    C:\..\> cd out
    C:\..\out> cmake -DUSER_LIB_PATH=C:\path\to\mingw64\lib\libpthreads.a -G "MinGW Makefiles" ..
    C:\..\out> make

As you can see, the Windows cmake command invocation is slightly different from
the Linux and Mac version, this is because we need to provide CMake with some
more info about where to find libphreads.a for later building. Furthermore,
we are forcing CMake to generate MingGW makefiles, because we don't support
other toolchain at the moment.

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

    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/test_apps.py

Additionally, an environment variable ``SKIP_ONLINE_TESTS`` can be used for
toggling the execution of the tests that require network access to the API.

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


Good first contributions
~~~~~~~~~~~~~~~~~~~~~~~~

You are welcome to contribute wherever in the code you want to, of course, but
we recommend taking a look at the "Good first contribution" label into the issues and
pick one. We would love to mentor you!

Doc
~~~

Review the parts of the documentation regarding the new changes and
update it if it's needed.

Pull requests
~~~~~~~~~~~~~

We use `GitHub pull requests
<https://help.github.com/articles/about-pull-requests>`_ to accept the
contributions.

A friendly reminder! We'd love to have a previous discussion about the best way to
implement the feature/bug you are contributing with. This is a good way to
improve code quality in our beloved SDK!, so remember to file a new Issue before
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
   without waiting to the big and final solution... otherwise, it would make
   reviewing and coming changes pretty difficult to accomplish. The reviewer
   will remove the **[WIP]** prefix from the description once the PR is ready
   to merge.

Please follow the next rules for the commit messages:

-  It should include a reference to the issue ID in the first line of the
   commit, **and** a brief description of the issue, so everybody knows what
   this ID actually refers to without wasting to much time on following the
   link to the issue.

-  It should provide enough information for a reviewer to understand the
   changes and their relation to the rest of the code.

A good example:

.. code::

    Issue #190: Short summary of the issue
    * One of the important changes
    * Another important change

A (really) bad example:

.. code::

    Fixes #190

Documentation
-------------

The documentation for the project is in the ``doc`` directory. The
documentation for the python SDK is auto-generated from python
docstrings using `Sphinx <http://www.sphinx-doc.org>`_ for generating the
documentation. Please follow `Google's Python Style
Guide <https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments>`_
for docstrings. A good example of the style can also be found with
`sphinx's napolean converter
documentation <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
You can see the rendered documentation for the stable version of the SDK at
`the SDK's landing page <https://qiskit.org/documentation>`_.

To generate the documentation, we need to invoke CMake first in order to generate
all specific files for our current platform.

See the previous *Building* section for details on how to run CMake.
Once CMake is invoked, all configuration files are in place, so we can build the
documentation running this command:

All platforms:

.. code:: sh

        $> cd out
        doc$> make doc
