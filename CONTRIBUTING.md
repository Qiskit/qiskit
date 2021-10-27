# Contributing

First read the overall project contributing guidelines. These are all
included in the qiskit documentation:

https://qiskit.org/documentation/contributing_to_qiskit.html

## Contributing to Qiskit Terra

In addition to the general guidelines there are specific details for
contributing to terra, these are documented below.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.

   If your code fails the local style checks (specifically the black
   code formatting check) you can use `tox -eblack` to automatically
   fix update the code formatting.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.

   If your pull request is adding a new class, function, or module that is
   intended to be user facing ensure that you've also added those to a
   documentation `autosummary` index to include it in the api documentation.
   For more details you can refer to:

   https://qiskit.org/documentation/contributing_to_qiskit.html#documentation-structure


3. If it makes sense for your change that you have added new tests that
   cover the changes.
4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a reno release note for that
   change and that the PR is tagged for the changelog.

### Changelog generation

The changelog is automatically generated as part of the release process
automation. This works through a combination of the git log and the pull
request. When a release is tagged and pushed to github the release automation
bot looks at all commit messages from the git log for the release. It takes the
PR numbers from the git log (assuming a squash merge) and checks if that PR had
a `Changelog:` label on it. If there is a label it will add the git commit
message summary line from the git log for the release to the changelog.

If there are multiple `Changelog:` tags on a PR the git commit message summary
line from the git log will be used for each changelog category tagged.

The current categories for each label are as follows:

| PR Label               | Changelog Category |
| -----------------------|--------------------|
| Changelog: Deprecation | Deprecated         |
| Changelog: New Feature | Added              |
| Changelog: API Change  | Changed            |
| Changelog: Removal     | Removed            |
| Changelog: Bugfix      | Fixed              |

### Release Notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit-terra. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
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

##### Linking to issues

If you need to link to an issue or other github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit/qiskit-terra/issues/12345>` for more
    details.
```

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

#### Building release notes locally

Building The release notes are part of the standard qiskit-terra documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

## Installing Qiskit Terra from source
Please see the [Installing Qiskit Terra from
Source](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-terra-from-source)
section of the Qiskit documentation.

## Test

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

##### STDOUT/STDERR and logging capture

When running tests in parallel using `stestr` either via tox, the Makefile
(`make test_ci`), or in CI we set the env variable
`QISKIT_TEST_CAPTURE_STREAMS` which will capture any text written to stdout,
stderr, and log messages and add them as attachments to the tests run so
output can be associated with the test case it originated from. However, if
you run tests with `stestr` outside of these mechanisms by default the streams
are not captured. To enable stream capture just set the
`QISKIT_TEST_CAPTURE_STREAMS` env variable to `1`. If this environment
variable is set outside of running with `stestr` the streams (STDOUT, STDERR,
and logging) will still be captured but **not** displayed in the test runners
output. If you are using the stdlib unittest runner a similar result can be
accomplished by using the
[`--buffer`](https://docs.python.org/3/library/unittest.html#command-line-options)
option (e.g. `python -m unittest discover --buffer ./test/python`).

##### Test Skip Options

How and which tests are executed is controlled by an environment
variable, `QISKIT_TESTS`:

Option | Description | Default
------ | ----------- | -------
`run_slow` | It runs tests tagged as *slow*. | `False`

It is possible to provide more than one option separated with commas.

Alternatively, the `make test_ci` target can be used instead of
`make test` in order to run in a setup that replicates the configuration
we used in our CI systems more closely.

### Snapshot Testing for Visualizations

If you are working on code that makes changes to any matplotlib visualisations
you will need to check that your changes don't break any snapshot tests, and add
new tests where necessary. You can do this as follows:

1. Make sure you have pushed your latest changes to your remote branch.
2. Go to link: `https://mybinder.org/v2/gh/<github_user>/<repo>/<branch>?urlpath=apps/test/ipynb/mpl_tester.ipynb`. For example, if your GitHub username is `username`, your forked repo has the same name the original, and your branch is `my_awesome_new_feature`, you should visit https://mybinder.org/v2/gh/username/qiskit-terra/my_awesome_new_feature?urlpath=apps/test/ipynb/mpl_tester.ipynb.
This opens a Jupyter Notebook application running in the cloud that automatically runs
the snapshot tests (note this may take some time to finish loading).
3. Each test result provides a set of 3 images (left: reference image, middle: your test result, right: differences). In the list of tests the passed tests are collapsed and failed tests are expanded. If a test fails, you will see a situation like this:

   <img width="995" alt="Screenshot_2021-03-26_at_14 13 54" src="https://user-images.githubusercontent.com/23662430/112663508-d363e800-8e50-11eb-9478-6d665d0ff086.png">
4. Fix any broken tests. Working on code for one aspect of the visualisations
can sometimes result in minor changes elsewhere to spacing etc. In these cases
you just need to update the reference images as follows:
    - download the mismatched images (link at top of Jupyter Notebook output)
    - unzip the folder
    - copy and paste the new images into `qiskit-terra/test/ipynb/mpl/references`,
  replacing the existing reference images
    - add, commit and push your changes, then restart the Jupyter Notebook app in your browser. The
  tests should now pass.
5. Add new snapshot tests covering your new features, extensions, or bugfixes.
    - add your new snapshot tests to `test/ipynb/mpl/test_circuit_matplotlib_drawer.py`
    , where you can also find existing tests to use as a guide.
    - commit and push your changes, restart the Jupyter Notebook app in your browser.
    As this is the first time you run your new tests there won't be any reference
    images to compare to. Instead you should see an option in the list of tests
    to download the new images, like so:

    <img width="1002" alt="Screenshot_2021-03-26_at_15 38 31" src="https://user-images.githubusercontent.com/23662430/112665215-b9c3a000-8e52-11eb-89e7-b18550718522.png">

    - download the new images, then copy and paste into `qiskit-terra/test/ipynb/mpl/references`
    - add, commit and push your changes, restart the Jupyter Notebook app in your browser. The
    new tests should now pass.

Note: If you have run `test/ipynb/mpl_tester.ipynb` locally it is possible some file metadata has changed, **please do not commit and push changes to this file unless they were intentional**.

## Style and lint

Qiskit Terra uses 2 tools for verify code formatting and lint checking. The
first tool is [black](https://github.com/psf/black) which is a code formatting
tool that will automatically update the code formatting to a consistent style.
The second tool is [pylint](https://www.pylint.org/) which is a code linter
which does a deeper analysis of the Python code to find both style issues and
potential bugs and other common issues in Python.

You can check that your local modifications conform to the style rules
by running `tox -elint` which will run `black` and `pylint` to check the local
code formatting and lint. If black returns a code formatting error you can
run `tox -eblack` to automatically update the code formatting to conform to
the style. However, if `pylint` returns any error you will have to fix these
issues by manually updating your code.

Because `pylint` analysis can be slow, there is also a `tox -elint-incr` target, which only applies
`pylint` to files which have changed from the source github. On rare occasions this will miss some
issues that would have been caught by checking the complete source tree, but makes up for this by
being much faster (and those rare oversights will still be caught by the CI after you open a pull
request).

## Development Cycle

The development cycle for qiskit-terra is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes.

### Branches

* `main`:

The main branch is used for development of the next version of qiskit-terra.
It will be updated frequently and should not be considered stable. The API
can and will change on main as we introduce and refine new features.

* `stable/*` branches:
Branches under `stable/*` are used to maintain released versions of qiskit-terra.
It contains the version of the code corresponding to the latest release for
that minor version on pypi. For example, stable/0.8 contains the code for the
0.8.2 release on pypi. The API on these branches are stable and the only changes
merged to it are bugfixes.

### Release cycle

When it is time to release a new minor version of qiskit-terra we will:

1.  Create a new tag with the version number and push it to github
2.  Change the `main` version to the next release version.

The release automation processes will be triggered by the new tag and perform
the following steps:

1.  Create a stable branch for the new minor version from the release tag
    on the `main` branch
2.  Build and upload binary wheels to pypi
3.  Create a github release page with a generated changelog
4.  Generate a PR on the meta-repository to bump the terra version and
    meta-package version.

The `stable/*` branches should only receive changes in the form of bug
fixes.

## Adding deprecation warnings
The qiskit-terra code is part of Qiskit and, therefore, the [Qiskit Deprecation Policy](https://qiskit.org/documentation/contributing_to_qiskit.html#deprecation-policy) fully applies here. Additionally, qiskit-terra does not allow `DeprecationWarning`s in its testsuite. If you are deprecating code, you should add a test to use the new/non-deprecated method (most of the time based on the existing test of the deprecated method) and alter the existing test to check that the deprecated method still works as expected, [using `assertWarns`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertWarns). The `assertWarns` context will silence the deprecation warning while checking that it raises.

For example, if `Obj.method1` is being deprecated in favour of `Obj.method2`, the existing test (or tests) for `method1` might look like this:

```python
def test_method1(self):
   result = Obj.method1()
   self.assertEqual(result, <expected>)
```

Deprecating `method1` means that `Obj.method1()` now raises a deprecation warning and the test will not pass. The existing test should be updated and a new test added for `method2`:


```python
def test_method1_deprecated(self):
   with self.assertWarns(DeprecationWarning):
       result = Obj.method1()
   self.assertEqual(result, <expected>)

def test_method2(self):
   result = Obj.method2()
   self.assertEqual(result, <expected>)
```

`test_method1_deprecated` can be removed after `Obj.method1` is removed (following the [Qiskit Deprecation Policy](https://qiskit.org/documentation/contributing_to_qiskit.html#deprecation-policy)).

## Dealing with the git blame ignore list

In the qiskit-terra repository we maintain a list of commits for git blame
to ignore. This is mostly commits that are code style changes that don't
change the functionality but just change the code formatting (for example,
when we migrated to use black for code formatting). This file,
`.git-blame-ignore-revs` just contains a list of commit SHA1s you can tell git
to ignore when using the `git blame` command. This can be done one time
with something like

```
git blame --ignore-revs-file .git-blame-ignore-revs qiskit/version.py

```

from the root of the repository. If you'd like to enable this by default you
can update your local repository's configuration with:

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

which will update your local repositories configuration to use the ignore list
by default.
