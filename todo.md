# TODO

- [ ] **`scripts/dggs/aspect.zig`: use `c_allocator` (or an arena) for the
  per-solve allocations instead of `GeneralPurposeAllocator`.** The survey
  feeds `skar.solve` a GPA, whose per-cell allocation overhead dominates the
  run: solving 100k cells/system takes ~1.5–1.9 s, vs ~0.24–0.48 s for the
  identical cells through `skar_py` (which uses `std.heap.c_allocator` in its
  C ABI). Same compiled solver — the ~3.5–6× gap is purely the allocator.
  Switch to `std.heap.c_allocator` (link libc) or a per-input arena reset each
  cell. See the skar_py DGGS port for the comparison.

- [ ] **Try `f128` for the solver core.** Zig makes swapping the float type
  easy, and skar's core is linear algebra — no trig — so it should work
  cleanly. Verified the load-bearing builtins all work at `f128` in Zig (they
  go through compiler-rt soft-float; no hardware quad needed):
  `@sqrt`, `@mulAdd` (used 59× in `linalg.zig`), and `@log` all compile and
  give full ~34-digit precision. Note the core isn't strictly
  transcendental-free: `skar.zig:488` has a `@log` in the log-det / duality-gap
  computation — but `@log(f128)` works, so it's a non-issue (the historical
  Zig `f128` gaps are in `sin`/`cos`/etc., which skar doesn't use; that pain
  came from sibling trig-heavy code, not here).

  Benefit: `f128` crushes the gap floor (`O(κ·ε)` drops ~18 orders), so
  ill-conditioned cells that currently `did_not_converge` at strict tol would
  converge — retiring the `gap_tol = 1e-3` workaround in the DGGS survey. But
  for `f64`-precision inputs (e.g. DGGS cell vertices) the aspect-ratio *answer*
  stays input-limited; `f128` buys certification/convergence, not a more
  accurate number, unless inputs are higher precision too.

  Parameterize the float type rather than hard-swapping so `f64` stays the
  default.
