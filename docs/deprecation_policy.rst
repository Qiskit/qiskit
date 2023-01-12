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

- we must not remove or change code without active warnings for least three
  months or two complete version cycles;

- there must always be a way to achieve valid goals that does not issue any
  warnings;

- never assume that a function that isn't explicitly internal isn't in use;

- all deprecations, changes and removals are considered API changes, and can
  only occur in minor releases not patch releases, per the
  :ref:`stable branch policy <stable_branch_policy>`.

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


.. issuing-deprecation-warnings:

Issuing deprecation warnings
============================

The proper way to raise a deprecation warning is to use the ``warn`` function
from the `warnings module in the Python standard library
<https://docs.python.org/3/library/warnings.html>`__, using the category
``DeprecationWarning``.  For example::

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


.. testing-deprecated-functionality:

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
This can be done in the docstring for the function, method, or class that is being deprecated, by adding a `deprecated note
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-deprecated>`__ :

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


In particularly situation where a deprecation or change might be a major disruptor for users, a *migration guide* might be needed.
Once the migration guide is written and published, deprecation messages and documentation can link to it.

