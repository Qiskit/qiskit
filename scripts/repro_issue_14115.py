"""Reproduction script for Qiskit/qiskit#14115 (Phase II).

Demonstrates that CommutativeCancellation does not expose or forward
approximation_degree, while sibling passes and the Rust backend do.
"""

from __future__ import annotations

import inspect
import sys

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CommutativeCancellation, RemoveIdentityEquivalent
from qiskit._accelerate import commutation_cancellation


def main() -> int:
    print("=== Issue #14115 reproduction ===\n")

    cc_sig = inspect.signature(CommutativeCancellation.__init__)
    rie_sig = inspect.signature(RemoveIdentityEquivalent.__init__)

    print("1. API gap: CommutativeCancellation.__init__ parameters")
    print(f"   {list(cc_sig.parameters.keys())}")
    print("   Expected to include: approximation_degree")
    print(f"   Has approximation_degree: {'approximation_degree' in cc_sig.parameters}\n")

    print("2. Sibling pass for comparison: RemoveIdentityEquivalent.__init__")
    print(f"   {list(rie_sig.parameters.keys())}")
    print(f"   Has approximation_degree: {'approximation_degree' in rie_sig.parameters}\n")

    cc_run_src = inspect.getsource(CommutativeCancellation.run)
    passes_degree_in_run = "approximation_degree" in cc_run_src
    print("3. CommutativeCancellation.run() forwards approximation_degree:", passes_degree_in_run)
    print("   (Rust cancel_commutations accepts it; Python pass does not pass it.)\n")

    cancel_sig = inspect.signature(commutation_cancellation.cancel_commutations)
    print("4. Rust binding cancel_commutations parameters:")
    print(f"   {list(cancel_sig.parameters.keys())}\n")

    qc = QuantumCircuit(1)
    qc.rz(0.001, 0)
    qc.rz(-0.001, 0)
    dag_default = circuit_to_dag(qc)
    dag_low = circuit_to_dag(qc)

    cc = CommutativeCancellation()
    cc.run(dag_default)
    ops_after_default = len(list(dag_default.op_nodes()))

    # Direct Rust call with lower approximation degree (more aggressive grouping)
    commutation_cancellation.cancel_commutations(dag_low, cc._commutation_checker, [], 0.01)
    ops_after_low = len(list(dag_low.op_nodes()))

    print("5. Behavioral check on near-canceling RZ pair (angles 0.001, -0.001):")
    print(f"   Ops after CommutativeCancellation (default): {ops_after_default}")
    print(f"   Ops after cancel_commutations(..., approximation_degree=0.01): {ops_after_low}")
    print(
        "   If counts differ, the pass ignores user approximation_degree "
        "and always uses the Rust default (1.0).\n"
    )

    print("6. Preset wiring (inspect source repo, not runtime):")
    print("   qiskit/transpiler/preset_passmanagers/builtin_plugins.py")
    print("   - RemoveIdentityEquivalent(approximation_degree=pass_manager_config.approximation_degree)")
    print("   - CommutativeCancellation()  # no approximation_degree passed\n")

    print("Reproduction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
