.. _capi-circuit-drawer-config:

=====================
QkCircuitDrawerConfig
=====================

.. c:struct:: QkCircuitDrawerConfig

The configuration for :c:func:`qk_circuit_draw`.  Create it with
:c:func:`qk_circuit_drawer_config_new`, set the options you care about with the setters below, and
release it with :c:func:`qk_circuit_drawer_config_free`.  Passing ``NULL`` to
:c:func:`qk_circuit_draw` in place of a configuration uses the same defaults as a freshly
constructed one.

The available options, with their defaults, are:

``bundle_cregs`` (default ``true``)
    If ``true``, bundles classical registers into single wires.  Set with
    :c:func:`qk_circuit_drawer_config_set_bundle_cregs`.

``merge_wires`` (default ``true``)
    If ``true``, merges the bottom and top lines of adjacent wires.  Set with
    :c:func:`qk_circuit_drawer_config_set_merge_wires`.

``fold`` (default ``0``)
    Sets the line length for wrapping the rendered text.  Use ``0`` to auto-detect the console
    width, and ``SIZE_MAX`` to effectively skip wrapping altogether.  Set with
    :c:func:`qk_circuit_drawer_config_set_fold`.

``barrier_label_len`` (default ``0``)
    Sets the number of characters to display for barrier labels.  If this number is exceeded, the
    label is truncated at that number and ``...`` is appended.  Use ``0`` to apply the default of
    16 characters.  Set with :c:func:`qk_circuit_drawer_config_set_barrier_label_len`.

Functions
=========

.. doxygengroup:: QkCircuitDrawerConfig
   :members:
   :content-only:
