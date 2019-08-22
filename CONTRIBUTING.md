# Contributing

## Contributing to Qiskit Terra

### Issue reporting

When you encounter a problem please open an issue for it to
the [issue tracker](https://github.com/Qiskit/qiskit-terra/issues).

### Improvement proposal

If you have an idea for a new feature please open an **Enhancement** issue in
the [issue tracker](https://github.com/Qiskit/qiskit-terra/issues). Opening an
issue starts a discussion with the team about your idea, how it fits in with
the project, how it can be implemented, etc.

### Code Review

Code review is done in the open and open to anyone. While only maintainers have
access to merge commits, providing feedback on pull requests is very valuable
and helpful. It is also a good mechanism to learn about the code base. You can
view a list of all open pull requests here:
https://github.com/Qiskit/qiskit-terra/pulls
to review any open pull requests and provide feedback on it.

### Good first contributions

If you would like to contribute to the qiskit-terra project, but aren't sure of where
to get started, the
[`good first issue`](
https://github.com/Qiskit/qiskit-terra/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
label highlights items for people new to the project to work on. These are all issues that have
been reviewed by contributors and tagged as something a new contributor
should be able to develop a fix for. In other words it shouldn't require
intimate familiarity with qiskit-terra to develop a fix for the issue.

### Documentation

If you make a change, make sure you update the associated
*docstrings* and parts of the
[documentation](https://github.com/Qiskit/qiskit/tree/master/docs/terra)
that corresponds to it. You can also make a [documentation issue](
https://github.com/Qiskit/qiskit/issues/new/choose) if you see doc bugs, have a
new feature that needs to be documented, or think that material could be added
to the existing docs.

### Pull requests

We use [GitHub pull requests](
https://help.github.com/articles/about-pull-requests) to accept contributions.

While not required, opening a new issue about the bug you're fixing or the
feature you're working on before you open a pull request is an important step
in starting a discussion with the community about your work. The issue gives us
a place to talk about the idea and how we can work together to implement it in
the code. It also lets the community know what you're working on and if you
need help, you can use the issue to go through it with other community and team
members.

If you've written some code but need help finishing it, want to get initial
feedback on it prior to finishing it, or want to share it and discuss prior
to finishing the implementation you can open a *Work in Progress* pull request.
When you create the pull request prefix the title with the **\[WIP\]** tag (for
**W**ork **I**n **P**rogress). This will indicate to reviewers that the code in
the PR isn't in it's final state and will change. It also means that we will
not merge the commit until it is finished. You or a reviewer can remove the
[WIP] tag when the code is ready to be fully reviewed for merging.

### Contributor License Agreement

Before you can submit any code we need all contributors to sign a
contributor license agreement. By signing a contributor license
agreement (CLA) you're basically just attesting to the fact
that you are the author of the contribution and that you're freely
contributing it under the terms of the Apache-2.0 license.

When you contribute to the Qiskit Terra project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The [individual CLA](https://qiskit.org/license/qiskit-cla.pdf)
document is available for review as a PDF.

**Note**:
> If your contribution is part of your employment or your contribution
> is the property of your employer, then you will likely need to sign a
> [corporate CLA](https://qiskit.org/license/qiskit-corporate-cla.pdf) too and
> email it to us at <qiskit@us.ibm.com>.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
3. If it makes sense for your change that you have added new tests that
   cover the changes.
4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have updated the CHANGELOG.md and
   added a reno release note for that change.

### Commit messages

As important as the content of the change, is the content of the commit message
describing it. The commit message provides the context for not only code review
but also the change history in the git log. Having a detailed commit message
will make it easier for your code to be reviewed and also provide context to the
change when it's being looked at years in the future. When writing a commit
message there are some important things to remember:

* Do not assume the reviewer understands what the original problem was.

When reading an issue, after a number of back & forth comments, it is often
clear what the root cause problem is. The commit message should have a clear
statement as to what the original problem is. The bug is merely interesting
historical background on *how* the problem was identified. It should be
possible to review a proposed patch for correctness from the commit message,
 without needing to read the bug ticket.
bug ticket.

* Do not assume the code is self-evident/self-documenting.

What is self-evident to one person, might not be clear to another person. Always
document what the original problem was and how it is being fixed, for any change
except the most obvious typos, or whitespace only commits.

* Describe why a change is being made.

A common mistake is to just document how the code has been written, without
describing *why* the developer chose to do it that way. By all means describe
the overall code structure, particularly for large changes, but more importantly
describe the intent/motivation behind the changes.

* Read the commit message to see if it hints at improved code structure.

Often when describing a large commit message, it becomes obvious that a commit
should have in fact been split into 2 or more parts. Don't be afraid to go back
and rebase the change to split it up into separate pull requests.

* Ensure sufficient information to decide whether to review.

When Github sends out email alerts for new pull request submissions, there is
minimal information included, usually just the commit message and the list of
files changes. Because of the high volume of patches, commit message must
contain sufficient information for potential reviewers to find the patch that
they need to look at.

* The first commit line is the most important.

In Git commits, the first line of the commit message has special significance.
It is used as the default pull request title, email notification subject line,
git annotate messages, gitk viewer annotations, merge commit messages, and many
more places where space is at a premium. As well as summarizing the change
itself, it should take care to detail what part of the code is affected.

* Describe any limitations of the current code.

If the code being changed still has future scope for improvements, or any known
limitations, then mention these in the commit message. This demonstrates to the
reviewer that the broader picture has been considered and what tradeoffs have
been done in terms of short term goals vs. long term wishes.

* Include references to issues

If the commit fixes or is related to an issue make sure you annotate that in
the commit message. Using the syntax:

Fixes #1234

if it fixes the issue (github will close the issue when the PR merges).

The main rule to follow is:

The commit message must contain all the information required to fully
understand & review the patch for correctness. Less is not more.

### Release Notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit-terra. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledege of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses qiskit, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

#### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with::

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root::

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like::

```yaml
features:
  - |
    Introduced a new feature foo, that adds support for doing something to
    ``QuantumCircuit`` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The ``qiskit.QuantumCircuit`` module has a new method ``foo()``. This is
    the equivalent of calling the ``qiskit.foo()`` to do something to your
    QuantumCircuit. This is the equivalent of running ``qiskit.foo()`` on
    your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The ``qiskit.bar`` module has been deprecated and will be removed in a
    future release. Its sole function, ``foobar()`` has been superseded by the
    ``qiskit.foo()`` function which provides similar functionality but with
    more accurate results and better performance. You should update your calls
    ``qiskit.bar.foobar()`` calls to ``qiskit.foo()``.
```

You can also look at other release notes for other examples.

You can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc) to express what is being changed as
needed. In general you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

#### Generating the release notes

After release notes have been added if you want to see what the full output of
the release notes. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full terra release
notes for all releases (since we started using reno during 0.9) you just run::

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged::

    reno report --version 0.9.0

At release time ``reno report`` is used to generate the release notes for the
release and the output will be submitted as a pull request to the documentation
repository's [release notes file](
https://github.com/Qiskit/qiskit/blob/master/docs/release_notes.rst)


## Installing Qiskit Terra from source
Please see the [Installing Qiskit Terra from
Source](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-terra-from-source)
section of the Qiskit documentation.

### Test

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox
with pip: `pip install -U tox`. Tox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. Additionally, the
environment that tox sets up matches the CI environment more closely and it
runs the tests in parallel (resulting in much faster execution). To run tests
on all installed supported python versions and lint/style checks you can simply
run `tox`. Or if you just want to run the tests once run for a specific python
version: `tox -epy37` (or replace py37 with the python version you want to use,
py35 or py36).

If you just want to run a subset of tests you can pass a selection regex to
the test runner. For example, if you want to run all tests that have "dag" in
the test id you can run: `tox -epy37 -- dag`. You can pass arguments directly to
the test runner after the bare `--`. To see all the options on test selection
you can refer to the stestr manual:
https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
tox -epy37 -- -n test.python.test_examples
```
or to run the same module by path:

```
tox -epy37 -- -n test/python/test_examples.py
```
to run a class:

```
tox -epy37 -- -n test.python.test_examples.TestPythonExamples
```
to run a method:
```
tox -epy37 -- -n test.python.test_examples.TestPythonExamples.test_all_examples
```

Alternatively there is a makefile provided to run tests, however this
does not perform any environment setup. It also doesn't run tests in
parallel and doesn't provide an option to easily modify the tests run.
For executing the tests with the makefile, a `make test` target is available.
The execution of the tests (both via the make target and during manual
invocation) takes into account the `LOG_LEVEL` environment variable. If
present, a `.log` file will be created on the test directory with the
output of the log calls, which will also be printed to stdout. You can
adjust the verbosity via the content of that variable, for example:

Linux and Mac:

``` {.bash}
$ cd out
out$ LOG_LEVEL="DEBUG" ARGS="-V" make test
```

Windows:

``` {.bash}
$ cd out
C:\..\out> set LOG_LEVEL="DEBUG"
C:\..\out> set ARGS="-V"
C:\..\out> make test
```

For executing a simple python test manually, we don\'t need to change
the directory to `out`, just run this command:

Linux and Mac:

``` {.bash}
$ LOG_LEVEL=INFO python -m unittest test/python/circuit/test_circuit_operations.py
```

Windows:

``` {.bash}
C:\..\> set LOG_LEVEL="INFO"
C:\..\> python -m unittest test/python/circuit/test_circuit_operations.py
```

##### Online Tests

Some tests require that you an IBMQ account configured. By default these
tests are always skipped. If you want to run these tests locally please
go to this
[page](https://quantumexperience.ng.bluemix.net/qx/account/advanced) and
register an account. Then you can either set the credentials explicitly
with the `IBMQ_TOKEN` and `IBMQ_URL` environment variables to specify
the token and url respectively for the IBMQ service. Alternatively, if
you already have a single set of credentials configured in your
environment (using a `.qiskitrc`) then you can just set
`QISKIT_TESTS_USE_CREDENTIALS_FILE` to `1` and it will use that.

##### Test Skip Options

How and which tests are executed is controlled by an environment
variable, `QISKIT_TESTS`:

Option | Description | Default
------ | ----------- | -------
`skip_online` | Skips tests that require remote requests. Does not require user credentials. | `False`
`run_slow` | It runs tests tagged as *slow*. | `False`

It is possible to provide more than one option separated with commas.

Alternatively, the `make test_ci` target can be used instead of
`make test` in order to run in a setup that replicates the configuration
we used in our CI systems more closely.

### Style guide

To enforce a consistent code style in the project we use
[Pylint](https://www.pylint.org) and
[pycodesytle](https://pycodestyle.readthedocs.io/en/latest/)
to verify that code contributions conform respect the projects
style guide. To verify that your changes conform to the style
guide you can run: `tox -elint`

Or using the makefile you can run:
```
make style
make lint
```

### Documentation

The documentation for Qiskit Terra is in the `docs` directory of [Qiskit
repository](https://github.com/Qiskit/qiskit/tree/master/docs). See this
repository for more information, however, the reference documentation is
auto-generated from the python docstrings throughout the code using
[Sphinx](http://www.sphinx-doc.org). Please follow [Google\'s Python
Style
Guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments)
for docstrings. A good example of the style can also be found with
[sphinx\'s napolean converter
documentation](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

### Development Cycle

The development cycle for qiskit-terra is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes and Changelog.

### Branches

* `master`:

The master branch is used for development of the next version of qiskit-terra.
It will be updated frequently and should not be considered stable. The API
can and will change on master as we introduce and refine new features.

* `stable/*` branches:
Branches under `stable/*` are used to maintain released versions of qiskit-terra.
It contains the version of the code corresponding to the latest release for
that minor version on pypi. For example, stable/0.8 contains the code for the
0.8.2 release on pypi. The API on these branches are stable and the only changes
merged to it are bugfixes.

### Release cycle

When it is time to release a new minor version of qiskit-terra we will:

1.  Create a stable branch for the new minor version from the current
    HEAD on the `master` branch
2.  Create a new tag with the version number on the HEAD of the new stable
    branch.
3.  Change the `master` version to the next release version.

The `stable/*` branches should only receive changes in the form of bug
fixes.

## Stable Branch Policy

The stable branch is intended to be a safe source of fixes for high
impact bugs and security issues which have been fixed on master since a
release. When reviewing a stable branch PR we need to balance the risk
of any given patch with the value that it will provide to users of the
stable branch. Only a limited class of changes are appropriate for
inclusion on the stable branch. A large, risky patch for a major issue
might make sense. As might a trivial fix for a fairly obscure error
handling case. A number of factors must be weighed when considering a
change:

-   The risk of regression: even the tiniest changes carry some risk of
    breaking something and we really want to avoid regressions on the
    stable branch
-   The user visible benefit: are we fixing something that users might
    actually notice and, if so, how important is it?
-   How self-contained the fix is: if it fixes a significant issue but
    also refactors a lot of code, it's probably worth thinking about
    what a less risky fix might look like
-   Whether the fix is already on master: a change must be a backport of
    a change already merged onto master, unless the change simply does
    not make sense on master.

### Backporting procedure:

When backporting a patch from master to stable we want to keep a
reference to the change on master. When you create the branch for the
stable PR you can use:

```
$ git cherry-pick -x $master_commit_id
```

However, this only works for small self contained patches from master.
If you need to backport a subset of a larger commit (from a squashed PR
for example) from master this just need be done manually. This should be
handled by adding:

    Backported from: #master pr number

in these cases, so we can track the source of the change subset even if
a strict cherry pick doesn\'t make sense.

If the patch you're proposing will not cherry-pick cleanly, you can help
by resolving the conflicts yourself and proposing the resulting patch.
Please keep Conflicts lines in the commit message to help review of the
stable patch.

### Backport Tags

Bugs or PRs tagged with `stable backport potential` are bugs
which apply to the stable release too and may be suitable for
backporting once a fix lands in master. Once the backport has been
proposed, the tag should be removed.

The PR against the stable branch should include `[Stable]`
in the title, as a sign that setting the target branch as stable was not
a mistake. Also, reference to the PR number in master that you are
porting.
