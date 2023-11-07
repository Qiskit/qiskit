#############################
Versioning and Support Policy
#############################

Qiskit version numbers follow `Sematinc Versioning <https://semver.org/>`__.
The version number is comprised of 3 primary components, the major, minor, and
patch versions. For a version number ``X.Y.Z`` where ``X`` is the major version,
``Y`` is the minor version, and ``Z`` is the patch version.

Breaking API changes are reserved for major version releases. The **minimum**
period between major version releases is one year. Minor versions will be
periodically (currently every three months) published for the current major
version which adds new features and bug fixes. For the most recent minor version
there will also be new patch versions published as bugs are identified and fixed
on that release series.

For the purposes of semantic versioning, the Qiskit public API is considered
any documented module, class, function, or method that is not marked as private
(with a ``_`` prefix). The supported Python versions, minimum supported Rust
version (for building Qiskit from source), and any depedency Python packages
(including the minimum supported versions of dependencies) used by Qiskit are
not part of the backwards compatibility guarantees and may change during any
release. Only minor or major version releases will raise minimum requirements
for using or building Qiskit (including adding new dependencies), but patch
fixes might include support for new versions of Python or other dependencies.

<TODO Add calendar diagrams>

With the release of a new major version, the previous major version is supported
for at least 6 months; only bug and security fixes will be accepted during this
time and only patch releases will be published for this major version. A final
patch version will be published when support is dropped and that release will
also document the end of support for that major version series. A longer
support window is needed for the previous major version as this gives downstream
consumers of Qiskit a chance to migrate not only their code but also their
users. It's typically not acceptable for a downstream library maintainer that
depends on Qiskit to immediately bump their minimum Qiskit version to a new
major version release immediately because their user base also needs a chance
to migrate to handle the API changes. By having an extended support window
for the previous major version that gives downstream maintainers to fix
compatibility the next major version but potentially keeping support for > 1
release series at a time and giving their users a migration path.

Upgrade Strategy
================

Whenever a new major version is released the recommended upgrade path
is to first upgrade to use the most recent minor version on the previous major
version. Immediately preceding a new major version a final minor version will
be published. This final minor version release ``0.N+1.0`` is equivalent to
``0.N.0`` but with warnings and deprecations for any API changes that are
made on the new major version series.

For example, on the release of Qiskit 1.0.0 a 0.46.0 release was published
immediately proceeding the 1.0.0 release. The 0.46.0 release was equivalent
to the 0.45.0 release but with additional deprecation warnings that documents
the API changes that are being made as part of the 1.0.0 release. This pattern
will be used for any future major version releases.

As a user of Qiskit it's recommended that you first upgrade to this final minor
version first, so you can see any deprecation warnings and adjust your Qiskit
usage ahead of time before trying a potentially breaking release. The previous
major version will be supported for at least 6 months to give sufficient time
to upgrade. A typical pattern to deal with this is to pin the max version to
avoid using the next major release series until you're sure of compatibility.
For example, specifying in a requirements file ``qiskit<2`` will ensure that
you're using a version of Qiskit that won't have breaking API changes.

Pre-releases
============

For each minor and major version release Qiskit will publish pre-releases that
are compatible with `PEP440 <https://peps.python.org/pep-0440/>`__. Typically
these are just release candidates of the form ``1.2.0rc1``. The ``rc`` will have
a finalized API surface and are used to test a prospective release.

If another PEP440 pre-release suffix (such as ``a``, ``b``, or ``pre``) are
published these do not have the same guarantees as an ``rc`` release, and are
just preview releases. The API likely will change between these pre-releases
and the final release with that version number. For example, ``1.0.0pre1`` has
a different final API from ``1.0.0``.

Post-releases
=============

If there are issues with the packaging of a given release a post-release may be
issued to correct this. These will follow the form ``1.2.1.1`` where the fourth
integer is used to indicate it is the 1st post release of the ``1.2.1`` release.
For example, the qiskit-terra (the legacy package name for Qiskit) 0.25.2
release had some issue with the sdist package publishing and a post-release
0.25.2.1 was published that corrected this issue. The code was identical, and
0.25.2.1 just fixed the packaging issue for the release.

##################
Deprecation Policy
##################

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
underscore (``_``).

The guiding principles are:

- we must not remove or change code without active warnings on a supported
  release series for at least three months and removals can only occur on
  major version releases;

- there must always be a way to achieve valid goals that does not issue any
  warnings;

- never assume that a function that isn't explicitly internal isn't in use;

- all deprecations can only occur in minor version releases not patch version
  releases, per the :ref:`stable branch policy <stable_branch_policy>`.

- API changes and removals are considered breaking changes, and can only
  occur in major version releases.

.. _removing-features:

Removing a feature
==================

When removing a feature (for example a class, function or function parameter),
we will follow this procedure:

#. The alternative path must be in place for one minor version before any
   warnings are issued.  For example, if we want to replace the function ``foo()``
   with ``bar()``, we must make at least one release with both functions before
   issuing any warnings within ``foo()``.  You may issue
   ``PendingDeprecationWarning``\ s from the old paths immediately.

   *Reason*: we need to give people time to swap over without breaking their
   code as soon as they upgrade.

#. After the alternative path has been in place for at least one minor version,
   :ref:`issue the deprecation warnings <issuing-deprecation-warnings>`.  Add a
   release note with a ``deprecations`` section listing all deprecated paths,
   their alternatives, and the reason for deprecation.  :ref:`Update the tests
   to test the warnings <testing-deprecated-functionality>`.

   *Reason*: removals must be highly visible for at least one version, to
   minimize the surprise to users when they actually go.

#. Set a removal date for the old feature, and remove it (and the warnings) when
   reached.  This must be at least three months after the version with the
   warnings was first released, and cannot be the minor version immediately
   after the warnings.  Add an ``upgrade`` release note that lists all the
   removals.  For example, if the alternative path was provided in ``0.19.0``
   and the warnings were added in ``0.20.0``, the earliest version for removal
   is ``0.22.0``, even if ``0.21.0`` was released more than three months after
   ``0.20.0``.

   .. note::

      These are *minimum* requirements.  For removal of significant or core features, give
      users at least an extra minor version if not longer.

   *Reason*: there needs to be time for users to see these messages, and to give
   them time to adjust.  Not all users will update their version of Qiskit
   immediately, and some may skip minor versions.

When a feature is marked as deprecated it is slated for removal, but users
should still be able to rely on it to work correctly.  We consider a feature
marked "deprecated" as frozen; we commit to maintaining it with critical bug
fixes until it is removed, but we won't merge new functionality to it.


Changing behavior
=================

Changing behavior without a removal is particularly difficult to manage, because
we need to have both options available for two versions, and be able to issue
warnings.  For example, changing the type of the return value from a function
will almost invariably involve making an API break, which is frustrating for
users and makes it difficult for them to use Qiskit.

The best solution here is often to make a new function, and then use :ref:`the
procedures for removal <removing-features>` above.

If you absolutely must change the behavior of existing code (other than fixing
bugs), you will need to use your best judgment to apply the guiding principles
at the top of this document.  The most appropriate warning for behavioral
changes is usually ``FutureWarning``.  Some possibilities for how to effect a
change:

- If you are changing the default behavior of a function, consider adding a
  keyword argument to select between old and new behaviors.  When it comes time,
  you can issue a ``FutureWarning`` if the keyword argument is not given
  (*e.g.* if it is ``None``), saying that the new value will soon become the
  default.  You will need to go through the normal deprecation period for
  removing this keyword argument after you have made the behavior change.  This
  will take at least six months to go through both cycles.

- If you need to change the return type of a function, consider adding a new
  function that returns the new type, and then follow the procedures for
  deprecating the old function.

- If you need to accept a new input that you cannot distinguish from an existing
  possibility because of its type, consider letting it be passed by a different
  keyword argument, or add a second function that only accepts the new form.


.. _issuing-deprecation-warnings:

Issuing deprecation warnings
============================

The proper way to raise a deprecation warning is to use the decorators ``@deprecate_arg`` and
``@deprecate_func`` from ``qiskit.utils.deprecation``. These will generate a standardized message and
and add the deprecation to that function's docstring so that it shows up in the docs.

.. code-block:: python

    from qiskit.utils.deprecation import deprecate_arg, deprecate_func

    @deprecate_func(since="0.24.0", additional_msg="No replacement is provided.")
    def deprecated_func():
        pass

    @deprecate_arg("bad_arg", new_alias="new_name", since="0.24.0")
    def another_func(bad_arg: str, new_name: str):
        pass

Usually, you should set ``additional_msg: str `` with the format ``"Instead, use ..."`` so that
people know how to migrate. Read those functions' docstrings for additional arguments like
``pending: bool`` and ``predicate``.

If you are deprecating outside the main Qiskit repo, set ``package_name`` to match your package.
Alternatively, if you prefer to use your own decorator helpers, then have them call
``add_deprecation_to_docstring`` from ``qiskit.utils.deprecation``.

If ``@deprecate_func`` and ``@deprecate_arg`` cannot handle your use case, consider improving
them. Otherwise, you can directly call the ``warn`` function
from the `warnings module in the Python standard library
<https://docs.python.org/3/library/warnings.html>`__, using the category
``DeprecationWarning``.  For example:

.. code-block:: python

    import warnings

    def deprecated_function():
        warnings.warn(
            "The function qiskit.deprecated_function() is deprecated since "
            "Qiskit Terra 0.20.0, and will be removed 3 months or more later. "
            "Instead, you should use qiskit.other_function().",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # ... the rest of the function ...

Make sure you include the version of the package that introduced the deprecation
warning (so maintainers can easily see when it is valid to remove it), and what
the alternative path is.

Take note of the ``stacklevel`` argument.  This controls which function is
accused of being deprecated.  Setting ``stacklevel=1`` (the default) means the
warning will blame the ``warn`` function itself, while ``stacklevel=2`` will
correctly blame the containing function.  It is unusual to set this to anything
other than ``2``, but can be useful if you use a helper function to issue the
same warning in multiple places.


.. _testing-deprecated-functionality:

Testing deprecated functionality
================================

Whenever you add deprecation warnings, you will need to update tests involving
the functionality.  The test suite should fail otherwise, because of the new
warnings.  We must continue to test deprecated functionality throughout the
deprecation period, to ensure that it still works.

To update the tests, you need to wrap each call of deprecated behavior in its
own assertion block.  For subclasses of ``unittest.TestCase`` (which all Qiskit
test cases are), this is done by:

.. code-block:: python

   class MyTestSuite(QiskitTestCase):
      def test_deprecated_function(self):
         with self.assertWarns(DeprecationWarning):
            output = deprecated_function()
         # ... do some things with output ...
         self.assertEqual(output, expected)

Documenting deprecations and breaking changes
=============================================

It is important to warn the user when your breaking changes are coming.

``@deprecate_arg`` and ``@deprecate_func`` will automatically add the deprecation to the docstring
for the function so that it shows up in docs.

If you are not using those decorators, you should directly add a `Sphinx deprecated directive
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-deprecated>`__:

.. code-block:: python

   def deprecated_function():
      """
      Short description of the deprecated function.

      .. deprecated:: 0.20.0
         The function qiskit.deprecated_function() is deprecated since
         Qiskit Terra 0.20.0, and will be removed 3 months or more later.
         Instead, you should use qiskit.other_function().

      <rest of the docstring>
      """
      # ... the rest of the function ...

You should also document the deprecation in the changelog by using Reno. Explain the deprecation
and how to migrate.

In particular situations where a deprecation or change might be a major disruptor for users, a
*migration guide* might be needed. Once the migration guide is written and published, deprecation
messages and documentation should link to it (use the ``additional_msg: str`` argument for
``@deprecate_arg`` and ``@deprecate_func``).
