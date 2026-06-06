---
features_transpiler:
  - |
    Added an ``approximation_degree`` argument to
    :class:`.CommutativeCancellation`. This float parameter (between 0 and 1,
    default ``1.0``) sets the tolerance threshold used when deciding whether
    two adjacent rotation angles are equivalent and can be cancelled. A value
    of ``1.0`` means only exact cancellations are performed. Values less than
    ``1.0`` allow approximate cancellation, which is useful when small
    floating-point errors would otherwise prevent the pass from recognising
    near-identical rotations.

    The ``approximation_degree`` argument accepted by
    :func:`.generate_preset_pass_manager` and :func:`.transpile` is now
    forwarded to :class:`.CommutativeCancellation` when building the
    preset optimization stage at all optimization levels that include this
    pass (levels 1, 2, and 3).
