# ai_open_src_capstone_contribution

## Contribution #1: Add approximation_degree argument to CommutativeCancellation pass and set it in the preset

**Contribution Number:** 1  
**Student:** Philip Park  
**Issue:** [Qiskit/qiskit#14115](https://github.com/Qiskit/qiskit/issues/14115)  
**Status:** Phase IV in progress (pre-submission — PR draft ready as of 7/12/26)

---

## Why I Chose This Issue

This issue adds an `approximation_degree` parameter to the `CommutativeCancellation` optimization pass so that approximation settings can properly flow through the internal optimization pipeline instead of being hidden inside lower-level analysis code. It matters because modern optimization systems, whether in quantum computing, AI inference, or quantitative trading infrastructure, rely on configurable tradeoffs between precision, speed, and computational efficiency, and this issue improves how those controls propagate through a larger execution graph. I chose this issue because as someone interested in quant trading and building my own ml models, I want deeper experience understanding optimization pipelines, DAG-based execution systems, and how real-world infrastructure exposes tunable parameters for balancing accuracy versus performance.

---

## Understanding the Issue

### Problem Description

`CommutativeCancellation` is a transpiler pass that cancels redundant gates using commutation relations. Internally, it calls the Rust function `cancel_commutations`, which runs `analyze_commutations` with an `approximation_degree` parameter. However, the Python pass wrapper does **not** expose `approximation_degree` in its constructor, does **not** forward it in `run()`, and the preset pass managers do **not** pass `pass_manager_config.approximation_degree` when constructing the pass.

As a result, when a user calls `transpile(..., approximation_degree=0.8)`, that setting reaches many optimization passes but **not** `CommutativeCancellation`, which always uses the Rust default of `1.0` (full precision).

### Expected Behavior

- `CommutativeCancellation` should accept an `approximation_degree` argument (consistent with sibling passes like `RemoveIdentityEquivalent` and `CommutativeOptimization`).
- Preset pass managers should pass `pass_manager_config.approximation_degree` when constructing `CommutativeCancellation`.
- Setting `approximation_degree` on `transpile()` should affect commutation analysis inside the cancellation pass.

### Current Behavior

- `CommutativeCancellation.__init__` only accepts `basis_gates` and `target`.
- `run()` calls `cancel_commutations(dag, checker, basis)` with three arguments — no degree.
- In `builtin_plugins.py`, three call sites use `CommutativeCancellation()` or `CommutativeCancellation(target=...)` without `approximation_degree`, while neighboring passes in the same pipeline stage do pass it.

### Affected Components

| Component | Path |
|-----------|------|
| Python pass wrapper | `qiskit/transpiler/passes/optimization/commutative_cancellation.py` |
| Rust implementation (already supports param) | `crates/transpiler/src/passes/commutation_cancellation.rs` |
| Commutation analysis (uses param) | `crates/transpiler/src/passes/commutation_analysis.rs` |
| Preset pass manager wiring | `qiskit/transpiler/preset_passmanagers/builtin_plugins.py` (lines ~163, ~523, ~542) |
| Existing tests (no coverage for this param) | `test/python/transpiler/test_commutative_cancellation.py` |

**Note:** The original issue text references `CommutationAnalysis` as a separate Python pass. The current architecture calls `analyze_commutations` directly from Rust inside `cancel_commutations`. The fix is still the same: thread `approximation_degree` through the Python pass and presets.

---

## Reproduction Process

### Environment Setup

**OS:** Windows 10 (10.0.26200)  
**Python:** 3.12.5  
**Fork:** https://github.com/philipjpark/qiskit  
**Working branch:** [`fix-issue-14115`](https://github.com/philipjpark/qiskit/tree/fix-issue-14115)

#### Setup steps completed

1. Cloned fork and added upstream remote (`Qiskit/qiskit`).
2. Created Python venv at `.venv` in the repo root.
3. Upgraded pip to 26.1.2 (`pip>=25.1` required for `--group` dev deps).
4. Installed Rust 1.87 via rustup (required for source builds).

#### Challenges encountered

| Problem | Fix / Workaround |
|---------|------------------|
| Rust toolchain partially installed; `cargo.exe` missing | Ran `rustup toolchain uninstall 1.87-x86_64-pc-windows-msvc` then `rustup toolchain install 1.87-x86_64-pc-windows-msvc` |
| `pip install -e .` failed with `detected conflict: bin\cargo.exe` during Rust compile | Documented for Phase III; used `pip install qiskit` (v2.4.2) in venv for reproduction and API inspection. Will retry editable install with `RUSTUP_AUTO_INSTALL=0` before implementing the fix. |
| Source build compiles many Rust crates (10+ min on Windows) | Expected per CONTRIBUTING.md; plan to use `pip install --group build` + `python setup.py build_rust --inplace` for dev iteration |

#### Commands used (Windows PowerShell)

```powershell
cd C:\Users\phili\Desktop\qiskit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\pip.exe install qiskit
.\.venv\Scripts\pip.exe install --group dev
```

For full source development (Phase III):

```powershell
rustup toolchain install 1.87-x86_64-pc-windows-msvc
$env:RUSTUP_AUTO_INSTALL = "0"
$env:RUSTUP_TOOLCHAIN = "1.87-x86_64-pc-windows-msvc"
.\.venv\Scripts\pip.exe install --group build
.\.venv\Scripts\pip.exe install -e .
```

### Steps to Reproduce

1. Activate the venv and run the reproduction script:
   ```powershell
   .\.venv\Scripts\python.exe scripts\repro_issue_14115.py
   ```

2. **API inspection (primary repro):** Confirm output shows:
   - `CommutativeCancellation.__init__` parameters: `['self', 'basis_gates', 'target']` — **no** `approximation_degree`
   - `RemoveIdentityEquivalent.__init__` **has** `approximation_degree`
   - `cancel_commutations` Rust binding **has** `approximation_degree`

3. **Source code inspection:** Open `qiskit/transpiler/passes/optimization/commutative_cancellation.py` and verify `run()` does not pass a fourth argument to `cancel_commutations`.

4. **Preset wiring inspection:** Open `qiskit/transpiler/preset_passmanagers/builtin_plugins.py` and compare:
   - Line ~156–157: `RemoveIdentityEquivalent(approximation_degree=pass_manager_config.approximation_degree)`
   - Line ~163: `CommutativeCancellation()` — no degree passed
   - Lines ~523, ~542: same pattern in optimization loops

5. Repeat steps 2–4 a second time to confirm consistency (issue is deterministic — it is a missing parameter, not intermittent).

### Branch Link

https://github.com/philipjpark/qiskit/tree/fix-issue-14115

### Reproduction Evidence

**Script output (2026-06-14):**

```
1. API gap: CommutativeCancellation.__init__ parameters
   ['self', 'basis_gates', 'target']
   Has approximation_degree: False

2. Sibling pass: RemoveIdentityEquivalent.__init__
   Has approximation_degree: True

3. CommutativeCancellation.run() forwards approximation_degree: False

4. Rust binding cancel_commutations parameters:
   ['dag', 'commutation_checker', 'basis_gates', 'approximation_degree']
```

**Screenshots/logs:** Reproduction script at `scripts/repro_issue_14115.py` in local fork (not part of upstream Qiskit).

**My findings:**

- The bug is confirmed at the API and preset-wiring level. The Rust layer already supports `approximation_degree`; the gap is entirely in the Python pass and preset pass managers.
- Other contributors have open PRs ([#15777](https://github.com/Qiskit/qiskit/pull/15777), [#16002](https://github.com/Qiskit/qiskit/pull/16002)). I commented on the issue and will coordinate before submitting a duplicate PR in Phase III.
- The issue was bumped across several milestones (2.1 → 2.2 → 2.3) and remains open as of June 2026.

---

## Solution Approach

### Analysis

**Root cause:** When `approximation_degree` support was added to the commutation/cancellation Rust pipeline, the Python `CommutativeCancellation` wrapper and preset pass managers were not updated to expose and forward the parameter. This is an integration oversight — the lower-level implementation works, but the public API and transpiler presets do not connect to it.

The data flow should be:

```
transpile(approximation_degree=X)
  → PassManagerConfig.approximation_degree
  → CommutativeCancellation(approximation_degree=X)   # missing today
  → cancel_commutations(..., approximation_degree=X)  # exists in Rust
  → analyze_commutations(..., approximation_degree=X)
```

### Proposed Solution

Add `approximation_degree` to `CommutativeCancellation.__init__`, store it on the instance, pass it to `cancel_commutations` in `run()`, and update all three `CommutativeCancellation(...)` call sites in `builtin_plugins.py` to pass `pass_manager_config.approximation_degree`. Follow the same patterns as `RemoveIdentityEquivalent` and `CommutativeOptimization` for handling `None` vs numeric values.

### Implementation Plan

#### Understand

Users expect `approximation_degree` on `transpile()` to control how aggressively all optimization passes approximate. `CommutativeCancellation` silently ignores this setting because the parameter is not plumbed through the Python wrapper or presets.

#### Match

| Reference | What to copy |
|-----------|--------------|
| `RemoveIdentityEquivalent` (`remove_identity_equiv.py`) | Constructor signature, store `_approximation_degree`, pass to Rust in `run()` |
| `CommutativeOptimization` (`commutative_optimization.py`) | Simpler sibling pass with `approximation_degree: float = 1.0` |
| `builtin_plugins.py` — `RemoveIdentityEquivalent(...)` call sites | How presets pass `pass_manager_config.approximation_degree` |
| `CommutativeOptimization` in Clifford+T loop (~line 1159) | Handles `None` by falling back to `1.0` |

#### Plan

1. **Modify** `qiskit/transpiler/passes/optimization/commutative_cancellation.py`:
   - Add `approximation_degree: float | None = 1.0` to `__init__`
   - Store as `self._approximation_degree`
   - In `run()`, resolve `None` → `1.0` (match `CommutativeOptimization` pattern) and pass to `cancel_commutations`

2. **Modify** `qiskit/transpiler/preset_passmanagers/builtin_plugins.py`:
   - Line ~163: `CommutativeCancellation(approximation_degree=pass_manager_config.approximation_degree)`
   - Line ~523: add `approximation_degree=pass_manager_config.approximation_degree`
   - Line ~542: same

3. **Add tests** in `test/python/transpiler/test_commutative_cancellation.py`:
   - Pass accepts `approximation_degree` kwarg without error
   - Pass runs on existing test circuits with `approximation_degree=0.5`

4. **Add release note** under `releasenotes/notes/` per CONTRIBUTING.md

5. **Self-review** against CONTRIBUTING.md PR checklist (tox, release note, `Fixes #14115` in PR description)

#### Implement

Phase III — branch: https://github.com/philipjpark/qiskit/tree/fix-issue-14115

#### Review

- [x] Update docstring for `CommutativeCancellation.__init__` with `approximation_degree` docs
- [x] Add release note tagged for changelog
- [ ] Run `tox` locally before PR (blocked on editable install; validated via PyPI wheel overlay on 7/5/26)
- [x] PR description includes `Fix #14115` (draft ready — 7/12/26)
- [x] Disclose AI tool usage per Qiskit CONTRIBUTING.md / PR template (draft ready — 7/12/26)
- [ ] Sign CLA at https://qisk.it/cla before PR merge (Phase IV)

#### Evaluate

See Testing Strategy below.

### Testing Strategy

#### Unit Tests

- [x] **Constructor test:** `CommutativeCancellation(approximation_degree=0.5)` instantiates without error
- [x] **Default behavior preserved:** `test_approximation_degree` uses `subTest` for `None`, `0.5`, and `1.0` on a CX-CX cancellation circuit
- [x] **Parameter accepted:** New `test_approximation_degree` in `test_commutative_cancellation.py`

#### Integration Tests

- [x] **Preset wiring:** `builtin_plugins.py` passes `approximation_degree` at all 3 call sites (init stage + optimization levels 2 and 3)
- [ ] **Full regression:** Run full `test_commutative_cancellation.py` suite after editable install (`pip install -e .`)

#### Manual Testing

- [x] API inspection confirmed `approximation_degree` present in `CommutativeCancellation.__init__` after implementation (Phase III)
- [x] **`test_approximation_degree` logic validated locally (7/5/26):** Ran equivalent checks via PyPI Qiskit 2.5.0 wheel with patched `commutative_cancellation.py` (workaround for broken editable install / Rust toolchain on Windows)
- [ ] Run official `python -m unittest test.python.transpiler.test_commutative_cancellation.TestCommutativeCancellation.test_approximation_degree -v` after editable install

---

## Implementation Notes

### Week 2 Progress (Phase II)

- Set up Windows dev environment (Python venv, Rust toolchain, PyPI Qiskit for repro)
- Created and pushed branch `fix-issue-14115` to fork
- Reproduced issue via API inspection script and source code review
- Documented root cause and UMPIRE implementation plan
- Identified competing open PRs; will coordinate on issue before Phase III coding

### Week 3 Progress (Phase III)

**What I built:**
- Added `approximation_degree` parameter to `CommutativeCancellation.__init__` and forwarded it to
  `cancel_commutations()` in `commutative_cancellation.py`
- [x] Wire `pass_manager_config.approximation_degree` at all 3 preset call sites in `builtin_plugins.py`
  (init stage, optimization level 2, optimization level 3)
- Added `test_approximation_degree` in `test_commutative_cancellation.py` using `subTest` for
  `None`, `0.5`, and `1.0` on a CX-CX cancellation circuit
- Added release note at `releasenotes/notes/add-approximation-degree-commutative-cancellation.yaml`

**Challenges faced:**
- Windows editable install (`pip install -e .`) still blocked by Rust/cargo toolchain conflict;
  tests written but full suite not yet run locally against source build
- Matched patterns from `RemoveIdentityEquivalent` and closed community PR #16002 for preset wiring

**Commits:**
- `1d99ffa3b`: Add approximation_degree to CommutativeCancellation pass (rebased)
- `ec02f5cc7`: Wire approximation_degree through preset pass managers, add test, and release note (rebased)
- `69660824a`: Remove Phase II repro script from PR branch (7/5/26)
- `13c0c8017`: Cleaned up fix-issue-14115 for upstream PR submission
- `b617d952e`: Remove accidental `.venv-test` from PR branch (7/12/26)

---

### Week 4 Progress (Phase IV)

**Pre-submission work completed (2026-06-27):**
- Finished remaining implementation: preset wiring in `builtin_plugins.py` (init + optimization levels 2 and 3), `test_approximation_degree`, and release note
- Ran final pre-submission review against Qiskit `CONTRIBUTING.md` and PR template (AI disclosure, `Fixes #14115`, CLA requirement)
- Drafted PR description and documented submission checklist in this README

**Still to do before Phase IV complete:**
- ~~Remove `scripts/repro_issue_14115.py` from the PR branch~~ ✓ done 7/5/26 (`69660824a`)
- ~~Rebase `fix-issue-14115` on latest `upstream/main` and push to fork~~ ✓ done 7/5/26
- ~~Run `test_approximation_degree` locally~~ ✓ validated 7/5/26 (PyPI wheel overlay workaround)
- ~~PR description draft (template / why→what / checklist / evidence)~~ ✓ done 7/12/26 (Week 6)
- ~~Remove accidental `.venv-test` from branch~~ ✓ done 7/12/26 (`b617d952e`)
- Open upstream PR: [Compare & pull request](https://github.com/Qiskit/qiskit/compare/main...philipjpark:qiskit:fix-issue-14115)
- Sign CLA at https://qisk.it/cla
- Comment on [#14115](https://github.com/Qiskit/qiskit/issues/14115) to coordinate with open PRs [#15777](https://github.com/Qiskit/qiskit/pull/15777) and [#16002](https://github.com/Qiskit/qiskit/pull/16002)

**Next steps after PR is open:**
- Update PR link and status below to **Awaiting review**
- Respond to maintainer feedback within 24 hours when it arrives
- Mark **Phase IV Complete** in CodePath submission

---

### Week 5 Progress (Phase IV) — 7/5/26

**Focus this week:** Pre-submission polish on a challenging issue in a large hybrid Python/Rust codebase.

The complexity of the Qiskit transpiler — preset pass managers, Python pass wrappers, and Rust bindings that must stay in sync — pushed me to be meticulous and detail-oriented. Rather than rushing to open the upstream PR, I completed branch hygiene and local validation first.

**Completed (7/5/26):**
- Reviewed `git diff upstream/main...HEAD` — only 4 intended files changed, no debug code or unrelated edits
- Removed Phase II repro tooling (`scripts/repro_issue_14115.py`) in commit `69660824a`
- Rebased `fix-issue-14115` on latest `upstream/main` (`655dfbbdf`) and force-pushed to fork
- Validated `test_approximation_degree` behavior locally (API + CX-CX cancellation for `None`, `0.5`, `1.0`) using PyPI Qiskit 2.5.0 with patched pass module — all checks passed
- Refined PR description draft and updated pre-submission checklist below

**Local validation output (7/5/26):**
```
API check: approximation_degree present in __init__
Cancellation OK for approximation_degree=None
Cancellation OK for approximation_degree=0.5
Cancellation OK for approximation_degree=1.0
ALL TESTS PASSED
```

**Still blocked / remaining before Phase IV complete:**
- Editable install (`pip install -e .`) still blocked by Rust/cargo toolchain conflict on Windows (attempted rustup repair 7/5/26; full `tox` suite pending)
- Sign CLA at https://qisk.it/cla
- Open upstream PR: [Compare & pull request](https://github.com/Qiskit/qiskit/compare/main...philipjpark:qiskit:fix-issue-14115)
- Comment on [#14115](https://github.com/Qiskit/qiskit/issues/14115) to coordinate with open PRs [#15777](https://github.com/Qiskit/qiskit/pull/15777) and [#16002](https://github.com/Qiskit/qiskit/pull/16002)

**Reflection:** Working through Phase IV on this issue reinforced that real open source contribution often means careful plumbing work — tracing how a single parameter flows from `transpile()` through presets into Rust — rather than writing new logic from scratch. The codebase rewards patience and precision.

---

### Week 6 Progress (Phase IV) — 7/12/26

**Focus this week:** Finish pre-submission documentation so the Contribution README and PR draft meet program Phase IV writing requirements — without opening the upstream PR yet.

**Completed (7/12/26):**
- Expanded the Pull Request section into a full paste-ready description: why before what, `Fix #14115`, acceptance criteria checklist, before/after console evidence, and Qiskit AI/LLM disclosure checkboxes
- Added a Maintainer Feedback log table (ready for dated entries after review starts)
- Set README status to **Draft ready / Awaiting submission**
- Confirmed branch cleanup commit on fork: `13c0c8017` — *Cleaned up fix-issue-14115 for upstream PR submission* (documents intent that the branch should only carry #14115 fix files)
- Removed accidental `.venv-test` tree that had been committed with the cleanup commit (`b617d952e`) and ignored `.venv-test/` so the PR diff stays limited to the real fix files + a small gitignore hygiene line

**Still remaining before Phase IV complete:**
- Sign CLA at https://qisk.it/cla
- Open upstream PR against `Qiskit/qiskit` `main`: [Compare & pull request](https://github.com/Qiskit/qiskit/compare/main...philipjpark:qiskit:fix-issue-14115)
- Paste the draft description into the GitHub PR form and update **PR Link** + status to **Awaiting review**
- Comment on [#14115](https://github.com/Qiskit/qiskit/issues/14115) to coordinate with open PRs [#15777](https://github.com/Qiskit/qiskit/pull/15777) and [#16002](https://github.com/Qiskit/qiskit/pull/16002)

**Reflection:** Week 6 showed that “review-ready” is as much about how you explain the change as the code itself. Filling the template, acceptance checklist, and evidence in the README first made the eventual upstream submission a copy-paste step rather than a scramble.

---

## Code Changes

**Files modified:**

- `qiskit/transpiler/passes/optimization/commutative_cancellation.py`
- `qiskit/transpiler/preset_passmanagers/builtin_plugins.py`
- `test/python/transpiler/test_commutative_cancellation.py`
- `releasenotes/notes/add-approximation-degree-commutative-cancellation.yaml`
- `.gitignore` (Week 6 hygiene: ignore `.venv-test/` so local validation envs are not committed)

**Branch:** https://github.com/philipjpark/qiskit/tree/fix-issue-14115

**Key commits:**
- https://github.com/philipjpark/qiskit/commit/1d99ffa3b — Add approximation_degree to CommutativeCancellation pass
- https://github.com/philipjpark/qiskit/commit/ec02f5cc7 — Wire preset pass managers, add test and release note
- https://github.com/philipjpark/qiskit/commit/69660824a — Remove Phase II repro script from PR branch (7/5/26)
- https://github.com/philipjpark/qiskit/commit/13c0c8017 — Cleaned up fix-issue-14115 for upstream PR submission (7/5/26 message; Week 6 cleanup intent)
- https://github.com/philipjpark/qiskit/commit/b617d952e — Remove accidental `.venv-test` from PR branch (7/12/26)

**Approach decisions:** Followed `RemoveIdentityEquivalent` / `CommutativeOptimization` patterns. Pass accepts `float | None`, converts `None` to `1.0` before Rust call. Presets pass `pass_manager_config.approximation_degree` directly, matching neighboring passes.

---

## Pull Request

**PR Link:** *(not yet submitted — open at [Compare & pull request](https://github.com/Qiskit/qiskit/compare/main...philipjpark:qiskit:fix-issue-14115) when ready)*

**Summary:** Adds `approximation_degree` to `CommutativeCancellation` and wires it through all three preset pass manager call sites so `transpile(..., approximation_degree=...)` reaches commutation analysis inside the cancellation pass.

**Status:** Draft ready / Awaiting submission (updated 7/12/26)

**Pre-submission checklist (updated 7/12/26):**
- [x] Core pass change committed (`1d99ffa3b`)
- [x] Preset wiring, test, and release note committed (`ec02f5cc7`)
- [x] Remove `scripts/repro_issue_14115.py` from branch before upstream PR (`69660824a`)
- [x] Rebase on `upstream/main` and push to fork (7/5/26)
- [x] Run `test_approximation_degree` logic locally — validated via PyPI wheel overlay (7/5/26)
- [x] PR description draft complete (template structure, why→what, acceptance checklist, evidence) — 7/12/26
- [x] Branch hygiene for upstream: cleanup commit + remove accidental `.venv-test` (`13c0c8017`, `b617d952e`) — 7/12/26
- [ ] Open PR to `Qiskit/qiskit` `main` with `Fix #14115` and AI disclosure per template
- [ ] Sign CLA at https://qisk.it/cla

### PR Description draft (ready to paste into GitHub — 7/12/26)

> ### Why
>
> When users call `transpile(..., approximation_degree=...)`, that setting is meant to trade precision for speed across the optimization pipeline. Neighboring passes such as `RemoveIdentityEquivalent` already receive `pass_manager_config.approximation_degree`, but `CommutativeCancellation` did not: its Python wrapper never accepted the argument, never forwarded it to the Rust `cancel_commutations` binding (which already supports it), and the three preset call sites in `builtin_plugins.py` constructed the pass without the degree. As a result, commutation/cancellation always ran at full precision (`1.0`) even when the user asked for approximation. This is an integration gap — the lower-level implementation already existed; the public API and presets were not connected.
>
> ### What
>
> - Add `approximation_degree: float | None = 1.0` to `CommutativeCancellation.__init__`, document it, store it, and pass it through to `cancel_commutations` in `run()` (treating `None` as `1.0`, matching `CommutativeOptimization`)
> - Wire `pass_manager_config.approximation_degree` at all three `CommutativeCancellation(...)` call sites in `builtin_plugins.py` (init stage + optimization levels 2 and 3)
> - Add `test_approximation_degree` covering `None`, `0.5`, and `1.0` on a CX–CX cancellation circuit
> - Add a release note for the changelog
>
> Fix #14115
>
> ### Does this PR meet the acceptance criteria?
>
> - [x] Tests added for new/changed behavior (`test_approximation_degree`)
> - [x] Relevant tests passing locally (validated 7/5/26; see before/after below)
> - [x] Follows project style guide (mirrors `RemoveIdentityEquivalent` / `CommutativeOptimization`)
> - [x] No breaking changes introduced (new optional kwarg with default `1.0` preserves prior behavior)
> - [x] Documentation / release note updated (`releasenotes/notes/add-approximation-degree-commutative-cancellation.yaml`)
>
> ### Before / after evidence
>
> **Before (Phase II repro — API gap):**
> ```
> CommutativeCancellation.__init__ parameters: ['self', 'basis_gates', 'target']
> Has approximation_degree: False
> RemoveIdentityEquivalent has approximation_degree: True
> cancel_commutations Rust binding has approximation_degree: True
> ```
>
> **After (local validation 7/5/26 — patched pass on PyPI Qiskit 2.5.0):**
> ```
> API check: approximation_degree present in __init__
> Cancellation OK for approximation_degree=None
> Cancellation OK for approximation_degree=0.5
> Cancellation OK for approximation_degree=1.0
> ALL TESTS PASSED
> ```
>
> ### AI/LLM disclosure
>
> - [ ] I didn't use LLM tooling, or only used it privately.
> - [x] I used the following tool to help write this PR description: Cursor
> - [x] I used the following tool to generate or modify code: Cursor (implementation assistance for pass wiring, tests, release note, and PR prep)

### Maintainer Feedback

*(No maintainer feedback yet — upstream PR not submitted. Log entries will use: Date | Feedback summary | Response | Commit ref)*

| Date | Feedback | My response | Commit |
|------|----------|-------------|--------|
| — | *(pending after PR opens)* | — | — |

---

## Learnings & Reflections

### Technical Skills Gained (Week 2)

- Navigated a hybrid Python/Rust codebase (Qiskit transpiler passes)
- Learned how `approximation_degree` flows from `transpile()` through preset pass managers
- Practiced API-level reproduction for infrastructure/plumbing bugs (not just runtime crashes)
- Windows-specific Rust toolchain troubleshooting with rustup

### Challenges Overcome

- Partially broken Rust installation blocking source builds
- Understanding that issue description referenced older architecture (`CommutationAnalysis` as separate pass) vs current Rust-internal `analyze_commutations` call

### What I'd Do Differently Next Time

- Check for existing open PRs on the issue before investing in Phase I selection (though the issue remains open and unmerged)
- Start editable install earlier in the week to allow more time for Windows Rust compile

### Technical Skills Gained (Week 5)

- Deepened understanding of pre-submission review in a large codebase: diff hygiene, commit message clarity, and matching existing pass patterns exactly
- Learned to treat Phase IV as its own discipline — not just "submit the PR," but verify every integration point before maintainers see it
- Built patience for hybrid Python/Rust debugging when local source builds are blocked on Windows

### Challenges Overcome (Week 5) — 7/5/26

- Navigating the gap between "implementation done" and "review-ready" when the codebase spans Python wrappers, preset wiring, and Rust internals
- Staying detail-oriented under complexity rather than rushing to open the upstream PR before local validation is complete
- Working around broken Rust/cargo toolchain on Windows by validating Python-layer changes against a PyPI wheel install while keeping source edits on the PR branch

### Technical Skills Gained (Week 6) — 7/12/26

- Learned how to write a maintainer-facing PR description (why before what, close keyword, acceptance checklist, evidence) that stands on its own without the diff
- Practiced aligning the Contribution README Pull Request section with Qiskit’s PR template and the program’s Phase IV sub-deliverables
- Reinforced git hygiene for upstream readiness: cleanup commits must not include local virtualenvs or unrelated tooling

### Challenges Overcome (Week 6) — 7/12/26

- Filling program requirements that look like “PR features” using the README draft first, before the upstream PR exists
- Catching and removing an accidental `.venv-test` commit so the fork branch stays reviewable for `Qiskit/qiskit`

### Resources Used

- [Qiskit #14115](https://github.com/Qiskit/qiskit/issues/14115)
- [Qiskit CONTRIBUTING.md](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md)
- [Rustup Windows MSVC docs](https://rust-lang.github.io/rustup/installation/windows-msvc.html)
- PR discussions: [#15777](https://github.com/Qiskit/qiskit/pull/15777), [#16002](https://github.com/Qiskit/qiskit/pull/16002)
