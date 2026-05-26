# Symmetry hygiene and structure-by-types opportunities

The FMA pass (`c901dc7`..`3f6e442`) tightened precision broadly and
pinned bit-exact symmetry of `Mat3.addSymRank1` / `Mat3.addSymRank2`
with regression tests. This doc collects the remaining
symmetry-related work — both small gaps the FMA pass missed and
larger structural improvements.

The codebase has several places where matrices are symmetric **by
construction** but stored as general `Mat3` (or `Mat2`) with the
symmetry maintained manually (compute upper triangle, mirror to
lower via exact assignment). The current pattern works, but it's
not structurally enforced — a future "tidy refactor" could
silently break it.

---

## A. Quick fixes

These are small, isolated, low-risk. Worth doing together as a
short batch.

### A1. FMA the `Mat2.addSymRank1` (missed in the original pass)

`src/linalg.zig:183` — same pattern as `Mat3.addSymRank1`, but
still uses old-style `+=` not `@mulAdd`:

```zig
pub inline fn addSymRank1(self: *Mat2, w: f64, p: Vec2) void {
    const wp0 = w * p.m[0];
    const wp1 = w * p.m[1];
    self.m[0] += wp0 * p.m[0];      // ← @mulAdd
    self.m[3] += wp1 * p.m[1];      // ← @mulAdd
    self.m[1] += wp0 * p.m[1];      // ← @mulAdd
    self.m[2] = self.m[1];
}
```

**Fix:** swap each `+=` for `@mulAdd(f64, wpj, p.m[k], self.m[i])`.
Multiplication order `(w·p_r)·p_c` preserved via the shared `wp_i`
precompute (same rationale as the Mat3 version, now documented in
its docstring).

**Where it runs:** `computeMoments` in `src/skar.zig:94` accumulates
the 2D moment matrix `M_s = Σ wᵢ·pᵢ·pᵢᵀ`, called every outer
iteration.

**Impact:** small per-call (Mat2 is tiny), but tight inner-loop site.
Same shape as the Mat3 win we already shipped.

---

## B. Defensive symmetry tests

Each of these methods produces bit-exact symmetric output by
construction. The asymmetry-could-leak-in risk is a future
refactor (someone "tidies up" by computing both halves with two
FMAs that round independently). Pin the contract with tests.

### B1. `Mat3.symmetrize` (`src/linalg.zig:422`)

Current impl: `return Mat3.lincomb(0.5, self, 0.5, self.transpose())`.
Bit-exact symmetric because:
- `0.5 * x` is exact for any finite `x` (just an exponent
  decrement, no rounding error)
- The cell (i,j) computes `0.5·m[i,j] + 0.5·m[j,i]`, cell (j,i)
  computes `0.5·m[j,i] + 0.5·m[i,j]` — equal under IEEE 754
  commutativity of addition

**Test shape:** start from a non-symmetric `Mat3.randomNormal`,
call `symmetrize`, assert `m[1]==m[3]`, `m[2]==m[6]`, `m[5]==m[7]`.

### B2. `Mat3.symOuter` (`src/linalg.zig:411`)

Current impl computes per cell `(x[i]·z[j] + z[i]·x[j]) * 0.5`.
Bit-exact symmetric: cell (i,j) and (j,i) compute the same sum
with operands swapped (commutative under IEEE 754).

**Test shape:** generate random `x, z`, call `symOuter`, assert
mirror equality.

### B3. `Mat2.addSymRank1` (`src/linalg.zig:183`)

Maintains `m[2] = m[1]` via exact assignment. Test that
`m[1] == m[2]` after the call.

### B4. mveeFw inline rank-1 (`src/skar.zig:204-225`)

The inlined accumulation that we FMA'd in commit `5994c09`
mirrors `m[3]=m[1]`, `m[6]=m[2]`, `m[7]=m[5]` after the loop.
Same invariant as `addSymRank1`, but the loop body could be
refactored separately from `addSymRank1` and drift.

**Test shape:** isn't directly testable as a unit (it's inline in
`mveeFw`). Two options:
- Extract a private helper `accumulateWeightedOuter` taking `Ql, w, *S`
  → testable + slightly more readable
- Add an invariant assertion (debug builds only) at the end of the
  inner loop verifying mirror equality — costs cycles but
  documents the contract

Option (a) probably preferred. Extracting is a small refactor.

**Notes:** All four tests would land in `tests/linalg_test.zig`
(extending the existing symmetry-invariant suite from
`tests/linalg_test.zig`'s addSymRank1/2 tests).

---

## C. Structural enforcement: `Sym3` / `Sym2` types

The big change. Replace ad-hoc symmetric `Mat3` / `Mat2` usage
with dedicated types that store only the upper triangle and
enforce symmetry at the type level.

### Sketch

```zig
pub const Sym3 = struct {
    // Upper triangle: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    m: [6]f64,

    pub const zero: Sym3 = .{ .m = .{0, 0, 0, 0, 0, 0} };

    // Symmetric rank-1 update (no mirror writes needed).
    pub inline fn addSymRank1(self: *Sym3, w: f64, q: Vec3) void { … }

    // Apply: y = M·x. Reads 6 entries with implicit symmetry.
    pub inline fn apply(self: Sym3, v: Vec3) Vec3 { … }

    // Cholesky on the upper triangle — matches what Mat3.cholesky
    // already does internally.
    pub fn cholesky(self: Sym3) ?Chol3 { … }

    // Eig: closed-form for 3×3 symmetric? Or refer to existing
    // eig2 for 2D shape.
    // Conversion to general Mat3 for places that need it.
    pub fn toMat3(self: Sym3) Mat3 { … }
};

pub const Sym2 = struct {
    // Upper triangle: (0,0), (0,1), (1,1)
    m: [3]f64,
    // ... analogous operations
};
```

### Wins

- **33% smaller memory:** `Sym3` is 48 bytes vs `Mat3`'s 72 bytes.
  Same hit rate impact in tight inner loops (mveeFw accumulates 6
  matrices per outer iter, summing in cache).
- **No mirror writes:** `addSymRank1` / `addSymRank2` shed the
  three `m[i] = m[j]` lines.
- **Symmetry guaranteed by the type system:** the regression tests
  added in commit `fed7637` become redundant (kept as
  documentation if useful, or deleted).
- **API clarity:** `cholesky(Sym3) → Chol3` is the signature that
  matches the math; mismatch with a general `Mat3` becomes a type
  error.
- **Forces explicit conversion when general operations needed:**
  `Sym3.toMat3()` calls become the marker for "I'm doing something
  general (matmul, transpose) on a structurally symmetric matrix"
  — easier to audit.

### Costs

- Real refactor. Sites touched:
  - `src/linalg.zig` — new `Sym3` / `Sym2` types, related ops
  - `src/skar.zig` — `mveeFw`'s S matrix, `computeMoments`'s
    `M_s`, `buildA`, `dualityGapConstructed`'s M, `recoverAPerp`
  - `src/newton.zig` — H matrix construction
  - `tests/linalg_test.zig` — symmetry tests redundant; could
    delete or move to Sym3 unit tests
- Mixed-type operations: `dualityGapConstructed` does `L^T · Z ·
  L` then symmetrizes. Z is `Mat3` (general). L is the Cholesky
  factor (lower triangular). The product is symmetric (matches
  Z's algebraic structure under SPD assumption). Would need a
  `Mat3.toSym3()` that runs `symmetrize` and casts, or a typed
  `congruence(L^T, Z): Sym3` helper.
- All test outputs (AR values, iter counts) should be unchanged —
  this is a pure type refactor, no FP changes. But it's a wide
  diff and worth landing on its own branch with careful review.

### Suggested ordering (if/when this lands)

1. Define `Sym3` (and `Sym2` if needed) alongside `Mat3` —
   coexistence first
2. Migrate `Mat3.addSymRank1` callers to `Sym3.addSymRank1` one at
   a time
3. Migrate the mveeFw inline accumulation (the biggest hot-loop
   user of symmetric `Mat3`)
4. Eventually: remove `Mat3.addSymRank1` and the symmetry tests
   from `tests/linalg_test.zig`

---

## Suggested priority

1. **A1** (Mat2.addSymRank1 FMA) — quick, fixes a real gap from
   the FMA pass
2. **B1-B3** (symmetry tests for symmetrize / symOuter / Mat2
   addSymRank1) — quick, defensive
3. **B4** (mveeFw inline test) — small refactor needed to make
   testable; do alongside the others or skip
4. **C** (Sym3/Sym2 types) — defer; chunky refactor, do when
   there's appetite for a structural change

A + B together is ~30 minutes of work and tightens the contract
across the remaining symmetric ops. C is a separate session.

---

## What's already protected

- `Mat3.addSymRank1` — FMA'd + bit-exact symmetry test (`fed7637`)
- `Mat3.addSymRank2` — same
- `Mat3.cholesky` reads only upper triangle (won't notice if lower
  drifts) — already-exploited symmetry
- Multiplication order documented inline (`dabcc9e`)

The Sym3/Sym2 path would generalize this protection structurally
instead of test-by-test.
