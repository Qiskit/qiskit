# Deprecation Policy

Starting from the 1.0.0 release, Qiskit follows semantic versioning, with a yearly release cycle for major releases.
[Full details of the scheduling are hosted with the external public documentation](https://docs.quantum.ibm.com/open-source/qiskit-sdk-version-strategy).

This document is primarily intended for developers of Qiskit themselves.

## Principles

Many users and other packages depend on different parts of Qiskit.  We must
make sure that whenever we make changes to the code, we give users ample time to
adjust without breaking code that they have already written.

Most importantly: *do not* change any interface that is public-facing unless we
absolutely have to.  Adding things is ok, taking things away is annoying for
users but can be handled reasonably with plenty notice, but changing behavior
generally means users cannot write code that will work with two subsequent
versions of Qiskit, which is not acceptable.

Beware that users will often be using functions, classes and methods that we,
the Qiskit developers, may consider internal or not widely used.  Do not make
assumptions that "this is buried, so nobody will be using it"; if it is public,
it is subject to the policy.  The only exceptions here are functions and modules
that are explicitly internal, *i.e.* those whose names begin with a leading
underscore (`_`).

The guiding principles are:

- removals or behavior changes in the public API can only occur in major releases;

- new deprecations to the public API can only occur in minor releases;

- there must always be a way to achieve valid goals that does not issue any
  warnings with the most recent two minor releases in a series;

- never assume that an object that is part of the public interface is not in use.

While the no-breaking-changes rule is only formally required *within* a major release series, you should make every effort to avoid breaking changes wherever possible.
Similarly, while it is permissible where necessary for behavior to change with no single-code path to support both the last minor of one major release and the first minor of a new major release, it is still strongly preferable if you can achieve this.


## What is the public interface?

> [!NOTE]
> This section should be in sync with [the release schedule documentation of Qiskit](https://docs.quantum.ibm.com/open-source/qiskit-sdk-version-strategy).
> Please [open an issue against Qiskit](https://github.com/Qiskit/qiskit/issues/new/choose) if there are discrepancies so we can clarify them.

For the purposes of semantic versioning, the Qiskit public API comprises all *publicly documented* packages, modules, classes, functions, methods, and attributes.

An object is *publicly documented* if and only if it appears in [the hosted API documentation](https://docs.quantum.ibm.com/api/qiskit) for Qiskit.
The presence of a docstring in the Python source (or a `__doc__` attribute) is not sufficient to make an object publicly documented; this documentation must also be rendered in the public API documentation.

As well as the objects themselves needing to be publicly documented, the only public-API *import locations* for a given object is the location it is documented at in [the public API documentation](https://docs.quantum.ibm.com/api/qiskit), and parent modules or packages that re-export the object (if any).
For example, while it is possible to import `Measure` from `qiskit.circuit.measure`, this is not a supported part of the public API for two reasons:

1. The module `qiskit.circuit.measure` is not publicly documented, so is not part of the public interface.
2. The [`Measure` object is documented as being in `qiskit.circuit.library`](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.Measure), and is re-exported by `qiskit.circuit`, so the public import paths are `from qiskit.circuit.library import Measure` and `from qiskit.circuit import Measure`.

As a rule of thumb, if you are using Qiskit, you should import objects from the highest-level package that exports that object.

Some components of the documented public interface may be marked as "experimental", and not subject to the stability guarantees of semantic versioning.
These will be clearly denoted in the documentation, and will raise an `ExperimentalWarning` when used.
We will only use these "experimental" features sparingly, when we feel there is a real benefit to making the experimental version public in an unstable form, such as a backwards-incompatible new version of core functionality that shows significant improvements over the existing form for limited inputs, but is not yet fully feature complete.
Typically, a feature will only become part of the public API when we are ready to commit to its stability properly.


## Removing a feature

> [!IMPORTANT]
> Features can only be removed in new major versions.
> Deprecations can only be added in new minor versions.

When removing a feature (for example a class, function or function parameter),
we will follow this procedure:

- The alternative path must be in place for one minor version before any
  warnings are issued.  For example, if we want to replace the function `foo()`
  with `bar()`, we must make at least one minor release with both functions before
  issuing any warnings within `foo()`.  You may issue
  `PendingDeprecationWarning`s from the old paths immediately, but this is not
  necessary and does not affect any timelines for removal.

  *Reason*: we need to give people time to swap over without breaking their
  code as soon as they upgrade.

- After the alternative path has been in place for at least one minor version,
  [issue the deprecation warnings](#issuing-deprecation-warnings).  Add a
  release note with a `deprecations` section listing all deprecated paths,
  their alternatives, and the reason for deprecation.  [Update the tests to test the warnings](#testing-deprecated-functionality).

  *Reason*: removals must be highly visible for at least one version, to
  minimize the surprise to users when they actually go.

- Apply the removal to the branch for the next major release, or open an issue to remind us to effect the removal and tag it for the milestone of the next major release.

> [!NOTE]
> These are _minimum_ requirements.
> For removal of significant or core features, try to give as long a warning period as is feasible.

When a feature is marked as deprecated it is slated for removal, but users
should still be able to rely on it to work correctly.  We consider a feature
marked "deprecated" as frozen; we commit to maintaining it with critical bug
fixes until it is removed, but we won't merge new functionality to it.


## Changing behavior

> [!IMPORTANT]
> Breaking behavior changes can only occur in new major versions, and should be avoided as much as possible.

Changing behavior without a removal is particularly difficult to manage, because
we need to have both options available for two versions, and be able to issue
warnings.  For example, changing the type of the return value from a function
will almost invariably involve making an API break, which is frustrating for
users and makes it difficult for them to use Qiskit.

The best solution here is often to make a new function, and then use [the procedures for removal](#removing-features) above.

If you absolutely must change the behavior of existing code (other than fixing
bugs), you will need to use your best judgment to apply the guiding principles
at the top of this document.  The most appropriate warning for behavioral
changes is usually `FutureWarning`.  Some possibilities for how to effect a
change:

- If you are changing the default behavior of a function, consider adding a
  keyword argument to select between old and new behaviors.  When it comes time,
  you can issue a `FutureWarning` if the keyword argument is not given
  (*e.g.* if it is `None`), saying that the new value will soon become the
  default.  You will need to go through the normal deprecation period for
  removing this keyword argument after you have made the behavior change.  This
  will take at least six months to go through both cycles.

- If you need to change the return type of a function, consider adding a new
  function that returns the new type, and then follow the procedures for
  deprecating the old function.

- If you need to accept a new input that you cannot distinguish from an existing
  possibility because of its type, consider letting it be passed by a different
  keyword argument, or add a second function that only accepts the new form.



## Issuing deprecation warnings

The proper way to raise a deprecation warning is to use the decorators `@deprecate_arg` and
`@deprecate_func` from `qiskit.utils.deprecation`. These will generate a standardized message and
and add the deprecation to that function's docstring so that it shows up in the docs.


```python
from qiskit.utils.deprecation import deprecate_arg, deprecate_func

@deprecate_func(since="0.24.0", additional_msg="No replacement is provided.")
def deprecated_func():
    pass

@deprecate_arg("bad_arg", new_alias="new_name", since="0.24.0")
def another_func(bad_arg: str, new_name: str):
    pass
```

Usually, you should set `additional_msg: str` with the format `"Instead, use ..."` so that
people know how to migrate. Read those functions' docstrings for additional arguments like
`pending: bool` and `predicate`.

If you are deprecating outside the main Qiskit repo, set `package_name` to match your package.
Alternatively, if you prefer to use your own decorator helpers, then have them call
`add_deprecation_to_docstring` from `qiskit.utils.deprecation`.

If `@deprecate_func` and `@deprecate_arg` cannot handle your use case, consider improving
them. Otherwise, you can directly call the `warn` function
from the [warnings module in the Python standard library](https://docs.python.org/3/library/warnings.html),
using the category `DeprecationWarning`.  For example:

```python
import warnings

def deprecated_function():
   warnings.warn(
      "The function qiskit.deprecated_function() is deprecated since "
      "Qiskit 0.44.0, and will be removed 3 months or more later. "
      "Instead, you should use qiskit.other_function().",
      category=DeprecationWarning,
      stacklevel=2,
   )
   # ... the rest of the function ...

```

Make sure you include the version of the package that introduced the deprecation
warning (so maintainers can easily see when it is valid to remove it), and what
the alternative path is.

Take note of the `stacklevel` argument.  This controls which function is
accused of being deprecated.  Setting `stacklevel=1` (the default) means the
warning will blame the `warn` function itself, while `stacklevel=2` will
correctly blame the containing function.  It is unusual to set this to anything
other than `2`, but can be useful if you use a helper function to issue the
same warning in multiple places.


## Testing deprecated functionality

Whenever you add deprecation warnings, you will need to update tests involving
the functionality.  The test suite should fail otherwise, because of the new
warnings.  We must continue to test deprecated functionality throughout the
deprecation period, to ensure that it still works.

To update the tests, you need to wrap each call of deprecated behavior in its
own assertion block.  For subclasses of `unittest.TestCase` (which all Qiskit
test cases are), this is done by:


```python
class MyTestSuite(QiskitTestCase):
   def test_deprecated_function(self):
      with self.assertWarns(DeprecationWarning):
         output = deprecated_function()
      # ... do some things with output ...
      self.assertEqual(output, expected)
```

## Documenting deprecations and breaking changes

It is important to warn the user when your breaking changes are coming.

`@deprecate_arg` and `@deprecate_func` will automatically add the deprecation to the docstring
for the function so that it shows up in docs.

If you are not using those decorators, you should directly add a [Sphinx deprecated directive](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-deprecated):


```python
def deprecated_function():
    """
    Short description of the deprecated function.

    .. deprecated:: 0.44.0
       The function qiskit.deprecated_function() is deprecated since
       Qiskit 0.44.0, and will be removed 3 months or more later.
       Instead, you should use qiskit.other_function().

    <rest of the docstring>
    """
    # ... the rest of the function ...
```


You should also document the deprecation in the changelog by using Reno. Explain the deprecation
and how to migrate.

In particular situations where a deprecation or change might be a major disruptor for users, a
*migration guide* might be needed. Please write these guides in Qiskit's documentation at
https://github.com/Qiskit/documentation/tree/main/docs/api/migration-guides. Once
the migration guide is written and published, deprecation
messages and documentation should link to it (use the `additional_msg` argument for
`@deprecate_arg` and `@deprecate_func`).
