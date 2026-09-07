#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compare QPY payload sizes between Qiskit builds.

Dumps a fixed set of circuits with :func:`qiskit.qpy.dump` at whatever QPY version the build calls
default, and prints a Markdown table -- the kind you paste into a PR description to show what a
change did to payload size.

Measurements go into a JSON store, keyed by the commit they were taken at, so one file accumulates
as many builds as you like::

    git checkout main && <rebuild>
    python tools/qpy_size.py qpy-sizes.json --record

    git checkout my-branch && <rebuild>
    python tools/qpy_size.py qpy-sizes.json --record --compare main my-branch

``--record`` measures the current build and inserts it.  A commit already in the store is left
alone -- pass ``--overwrite`` to replace it -- so re-running the record step costs nothing and
never silently rewrites history.  ``--compare`` takes the two revisions to compare and reads the
store only, so it needs no rebuild of either side.

Any table can also be written to a Markdown file with ``--out PATH.md``, which is printed to stdout
as well.

With no arguments the script prints the current build's sizes and records nothing.
"""

import argparse
import datetime
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import ClassicalRegister, Parameter, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit.transpiler import CouplingMap

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT = shutil.which("git")

# Bumped only if the layout of the store changes incompatibly, so a stale file fails loudly instead
# of being half-read.
SCHEMA_VERSION = 1

# Every circuit is given an explicit name.  An unnamed QuantumCircuit is called ``circuit-<N>`` after
# a process-global counter, and that name is written into the payload -- so its length, and hence the
# payload size, would drift with how many circuits the process built first.


def _bell():
    """Smallest interesting payload: fixed overhead dominates."""
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _random():
    """General instruction volume, where per-instruction cost dominates."""
    from qiskit.circuit.random import random_circuit

    qc = random_circuit(20, 256, max_operands=3, measure=True, reset=True, seed=42)
    qc.name = "random"
    return qc


def _parameterized():
    """Many references to a few symbols."""
    qc = QuantumCircuit(50, name="parameterized")
    params = [Parameter(f"angle{i}") for i in range(10)]
    for qubit in range(qc.num_qubits - 1):
        for param in params:
            qc.rx(param, qubit)
    return qc


def _custom_gates():
    """One custom definition referenced thousands of times."""
    inner = QuantumCircuit(2, name="custom_gate")
    inner.h(0)
    inner.x(1)
    gate = inner.to_gate()
    qc = QuantumCircuit(200, name="custom_gates")
    for _ in range(100):
        for qubit in range(qc.num_qubits - 1):
            qc.append(gate, [qubit, qubit + 1])
    return qc


def _control_flow():
    """if/else, while, for, switch and box."""
    qr, cr = QuantumRegister(3, "qr"), ClassicalRegister(3, "cr")
    qc = QuantumCircuit(qr, cr, name="control_flow")
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])
    with qc.if_test(expr.equal(cr, 3)) as else_:
        qc.x(qr[0])
    with else_:
        qc.z(qr[0])
    with qc.while_loop((cr, 1)):
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
    with qc.for_loop(range(3)) as i:
        qc.rz(i, qr[1])
    with qc.switch(cr) as case:
        with case(0):
            qc.x(qr[1])
        with case(case.DEFAULT):
            qc.y(qr[1])
    with qc.box(duration=13):
        qc.cx(qr[0], qr[2])
    return qc


def _pauli_evolution():
    """Carries an operator and its synthesis settings."""
    qc = QuantumCircuit(4, name="pauli_evolution")
    operator = SparsePauliOp.from_list([("ZZII", 1.0), ("IXXI", 0.5), ("IIYY", -0.25)])
    qc.append(PauliEvolutionGate(operator, time=2, synthesis=LieTrotter(reps=2)), range(4))
    return qc


def _transpiled():
    """A transpiled circuit, so the LAYOUT section is populated rather than skipped."""
    qc = QuantumCircuit(6, name="pre_transpile")
    qc.h(0)
    for qubit in range(5):
        qc.cx(0, qubit + 1)
    qc.measure_all()
    out = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=CouplingMap.from_line(6),
        optimization_level=1,
        seed_transpiler=42,
    )
    out.name = "transpiled"
    return out


CIRCUITS = {
    "bell": _bell,
    "random": _random,
    "parameterized": _parameterized,
    "custom_gates": _custom_gates,
    "control_flow": _control_flow,
    "pauli_evolution": _pauli_evolution,
    "transpiled": _transpiled,
}


def _git(*args):
    """Output of a git command in this checkout, or ``None`` if git cannot answer."""
    if GIT is None:
        return None
    try:
        result = subprocess.run(
            [GIT, *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _head():
    """Identity of the checked-out commit, or ``None`` outside a git checkout.

    ``dirty`` only counts modifications to tracked files: an untracked file is usually a scratch
    script rather than something the build picked up.
    """
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return None
    return {
        "commit": commit,
        "subject": _git("log", "-1", "--format=%s") or "",
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def measure():
    """Payload size of every circuit, plus the build that produced them.

    A circuit that this Qiskit cannot build or dump records ``None`` rather than aborting, so an
    older build still yields a usable table for everything else.
    """
    sizes = {}
    for name, build in CIRCUITS.items():
        try:
            buffer = io.BytesIO()
            qpy.dump(build(), buffer)
            sizes[name] = len(buffer.getvalue())
        except Exception as exc:  # pylint: disable=broad-except
            print(f"note: {name} unavailable on this build ({type(exc).__name__})", file=sys.stderr)
            sizes[name] = None
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "qiskit_version": qiskit.__version__,
        "qpy_version": qpy.common.QPY_VERSION,
        "sizes": sizes,
    }


def snapshot():
    """A measurement of this build, tagged with the commit it was taken at."""
    return {**(_head() or {}), **measure()}


def load_store(path):
    """The store at ``path``, or an empty one if the file does not exist yet."""
    if not path.exists():
        return {"schema": SCHEMA_VERSION, "runs": {}}
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(store, dict) or not isinstance(store.get("runs"), dict):
        raise SystemExit(f"{path} is not a qpy_size store (expected a top-level 'runs' object)")
    if store.get("schema") != SCHEMA_VERSION:
        raise SystemExit(
            f"{path} has schema {store.get('schema')!r}, but this script writes "
            f"schema {SCHEMA_VERSION}"
        )
    return store


def save_store(path, store):
    """Write ``store`` back to ``path``."""
    path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")


def record(store, overwrite=False):
    """Measure this build and file it in ``store`` under the checked-out commit.

    Returns the entry now held for that commit and whether the store changed.  The store is
    consulted *before* anything is measured, so re-running against a commit that is already
    recorded costs nothing -- that is what makes this safe to put in front of every comparison.
    """
    head = _head()
    if not head:
        raise SystemExit(
            "cannot record: no commit to key the run by (not a git checkout). Run without a store "
            "path to just print the sizes."
        )
    commit = head["commit"]
    if commit in store["runs"] and not overwrite:
        print(
            f"note: {commit[:9]} is already recorded ({_when(store['runs'][commit])}), so nothing "
            "was measured; pass --overwrite to replace it",
            file=sys.stderr,
        )
        return store["runs"][commit], False
    if head["dirty"]:
        print(
            f"warning: tracked files are modified, so {commit[:9]} does not describe what is being "
            "measured; the run is marked dirty",
            file=sys.stderr,
        )
    store["runs"][commit] = entry = {**head, **measure()}
    return entry, True


def resolve(store, rev):
    """The full commit key in ``store`` that ``rev`` names.

    Accepts a key, a unique prefix of one, or anything ``git rev-parse`` understands -- so
    ``--compare main my-branch`` works without pasting hashes.
    """
    runs = store["runs"]
    if rev in runs:
        return rev
    matches = sorted(commit for commit in runs if commit.startswith(rev))
    if len(matches) > 1:
        raise SystemExit(
            f"{rev!r} matches several recorded runs: " + ", ".join(c[:9] for c in matches)
        )
    if matches:
        return matches[0]
    resolved = _git("rev-parse", rev)
    if resolved and resolved in runs:
        return resolved
    known = ", ".join(commit[:9] for commit in runs) or "none"
    raise SystemExit(f"no run recorded for {rev!r} (recorded: {known})")


def _total(sizes):
    return sum(size or 0 for size in sizes.values())


def _pct(before, after):
    return "" if not before else f" ({(after - before) / before * 100:+.2f}%)"


def _when(entry):
    return entry.get("recorded_at", "unknown time")


def _md(text):
    """A commit subject is free text; a bare pipe in it would end the table cell early."""
    return text.replace("|", r"\|")


def _describe(entry):
    """One-line provenance for a run: commit, build, QPY version, when it was taken."""
    parts = []
    if entry.get("commit"):
        subject = entry.get("subject")
        parts.append(f"`{entry['commit'][:9]}`" + (f' "{_md(subject)}"' if subject else ""))
    parts.append(f"Qiskit `{entry['qiskit_version']}`")
    parts.append(f"QPY {entry['qpy_version']}")
    if entry.get("recorded_at"):
        parts.append(f"recorded {entry['recorded_at']}")
    if entry.get("dirty"):
        parts.append("**uncommitted changes**")
    return ", ".join(parts)


def render_run(entry):
    """Markdown table of one run's sizes."""
    lines = ["| circuit | bytes |", "| --- | ---: |"]
    for name, size in entry["sizes"].items():
        lines.append(f"| {name} | {'n/a' if size is None else format(size, ',')} |")
    lines.append(f"| **total** | **{_total(entry['sizes']):,}** |")
    lines.append("")
    lines.append(_describe(entry))
    return "\n".join(lines)


def render_comparison(before_run, after_run):
    """Markdown table of two runs side by side, with the change between them."""
    before_label = (before_run.get("commit") or "before")[:9]
    after_label = (after_run.get("commit") or "after")[:9]
    lines = [
        f"| circuit | before (`{before_label}`) | after (`{after_label}`) | change |",
        "| --- | ---: | ---: | ---: |",
    ]
    before_total = after_total = comparable = 0
    for name in after_run["sizes"]:
        before, after = before_run["sizes"].get(name), after_run["sizes"][name]
        if before is None or after is None:
            shown = ["n/a" if size is None else format(size, ",") for size in (before, after)]
            lines.append(f"| {name} | {shown[0]} | {shown[1]} | n/a |")
            continue
        # Only circuits measured on both sides go into the total; counting a missing one as zero
        # would make whichever build lacks it look smaller.
        comparable += 1
        before_total += before
        after_total += after
        change = "no change" if after == before else f"{after - before:+,}{_pct(before, after)}"
        lines.append(f"| {name} | {before:,} | {after:,} | {change} |")
    delta = after_total - before_total
    summary = "no change" if delta == 0 else f"{delta:+,}{_pct(before_total, after_total)}"
    lines.append(
        f"| **total** ({comparable} circuits) | **{before_total:,}** "
        f"| **{after_total:,}** | **{summary}** |"
    )
    lines.append("")
    lines.append(f"- before: {_describe(before_run)}")
    lines.append(f"- after: {_describe(after_run)}")
    if before_run["qpy_version"] != after_run["qpy_version"]:
        lines.append(
            f"- **The QPY version changed ({before_run['qpy_version']} to "
            f"{after_run['qpy_version']}), so these sizes are not like-for-like.**"
        )
    if before_run.get("dirty") or after_run.get("dirty"):
        lines.append(
            "- **A run marked with uncommitted changes was not measured at the commit it is "
            "filed under.**"
        )
    return "\n".join(lines)


def render_listing(store):
    """Markdown table of what a store holds, newest entry last."""
    if not store["runs"]:
        return "no runs recorded yet"
    lines = [
        "| commit | subject | qiskit | QPY | total bytes | recorded |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for commit, entry in store["runs"].items():
        dirty = " *(dirty)*" if entry.get("dirty") else ""
        lines.append(
            f"| `{commit[:9]}`{dirty} | {_md(entry.get('subject', ''))} "
            f"| `{entry['qiskit_version']}` | {entry['qpy_version']} "
            f"| {_total(entry['sizes']):,} | {_when(entry)} |"
        )
    return "\n".join(lines)


def emit(text, out=None):
    """Print a rendered table, and also write it to ``out`` if one was asked for.

    The table still goes to stdout either way, so piping keeps working; the file is overwritten,
    since it holds a rendered report rather than a log.
    """
    print(text)
    if out is None:
        return
    path = Path(out)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def main(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "examples:\n"
            "  qpy_size.py                                  print this build's sizes\n"
            "  qpy_size.py sizes.json --record              measure and add this commit\n"
            "  qpy_size.py sizes.json --record --overwrite  replace an existing entry\n"
            "  qpy_size.py sizes.json --compare main HEAD   table for a PR description\n"
            "  qpy_size.py sizes.json --list                what the store holds\n"
            "  qpy_size.py sizes.json --compare main HEAD --out sizes.md   ...and save it\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "store",
        nargs="?",
        metavar="STORE",
        help="JSON file of recorded runs, keyed by commit (created if absent)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="measure this build and insert it into STORE under the checked-out commit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="with --record, replace an entry that is already in STORE for this commit",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="print the change between two runs in STORE; each is a commit hash, a unique prefix, "
        "or any revision git understands (a branch, HEAD~1, ...)",
    )
    parser.add_argument("--list", action="store_true", dest="list_runs", help="list runs in STORE")
    parser.add_argument(
        "--out",
        metavar="PATH.md",
        help="also write the Markdown to PATH.md (overwriting it), for pasting or attaching",
    )
    args = parser.parse_args(argv)

    actions = [args.record, bool(args.compare), args.list_runs]
    if args.store is None:
        if any(actions) or args.overwrite:
            parser.error("--record, --compare, --overwrite and --list all need a STORE path")
        emit(render_run(snapshot()), args.out)
        return 0
    if not any(actions):
        parser.error(f"nothing to do with {args.store}: pass --record, --compare or --list")
    if args.overwrite and not args.record:
        parser.error("--overwrite only means something with --record")

    path = Path(args.store)
    if not path.exists() and not args.record:
        raise SystemExit(f"{path} does not exist yet -- create it with --record")
    store = load_store(path)
    recorded = None
    if args.record:
        recorded, changed = record(store, overwrite=args.overwrite)
        if changed:
            save_store(path, store)
            print(f"recorded {recorded['commit'][:9]} in {path}", file=sys.stderr)

    tables = []
    if args.compare:
        before, after = (resolve(store, rev) for rev in args.compare)
        tables.append(render_comparison(store["runs"][before], store["runs"][after]))
    if args.list_runs:
        tables.append(render_listing(store))
    if recorded is not None and not tables:
        tables.append(render_run(recorded))
    emit("\n\n".join(tables), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
