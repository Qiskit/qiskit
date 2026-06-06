---
features_transpiler:
  - |
    Added an ``approximation_degree`` argument to
    :class:`.CommutativeCancellation`. This float parameter (within ``(0, 1]``,
    default ``1.0``) controls the tolerance used when analyzing commutations
    between gates. The ``approximation_degree`` argument accepted by
    :func:`.generate_preset_pass_manager` and :func:`.transpile` is now
    forwarded to :class:`.CommutativeCancellation` when building the
    preset optimization stage at all optimization levels that include this
    pass (levels 1, 2, and 3).
