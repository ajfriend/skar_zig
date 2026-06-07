//! Internal configuration for the solver: structural geometric
//! constants (`SIGMA_0`), algorithm tuning knobs (`algo`), and
//! numerical-precision tolerances (`tol`). Not exposed to the public
//! API — user-tunable parameters live in `SolveOptions` in `skar.zig`.

/// Structural axial eigenvalue: A·b = SIGMA_0·b, where b is the cone axis.
/// Derived in `recoverAPerp` via the budget/g_max rescaling: λ_b = √(1 − 2/3).
/// Not tunable — it's geometry, not a knob.
pub const SIGMA_0: f64 = 1.0 / @sqrt(3.0);

/// Algorithm tuning parameters — internal knobs tuned together for the
/// solver to converge cleanly. Not exposed to callers because they
/// interact subtly: changing one without coordinated changes to others
/// can break convergence. Adjust here if you're working on the algorithm
/// itself; user-facing tuning is in `SolveOptions`.
pub const algo = struct {
    /// Number of (project + FW + b-update) cycles per outer iteration.
    /// Only the final cycle of each outer iteration runs Newton polish
    /// + gap check. FW_PER_NEWTON = 1 is the original behaviour.
    pub const FW_PER_NEWTON: u32 = 2;

    /// Damping curve for the b-update: shrink alpha when |c| grew,
    /// grow when |c| shrank, bounded in [DAMP_MIN, DAMP_MAX].
    pub const DAMP_SHRINK: f64 = 0.5;
    pub const DAMP_GROW: f64 = 1.2;
    pub const DAMP_MIN: f64 = 0.05;
    pub const DAMP_MAX: f64 = 1.0;

    /// Support-set membership cutoff: a point counts as active (kept in
    /// the `Info.cert`, in the constructed dual `Z`, and — load-bearing —
    /// in the Newton-polish active set) iff its weight exceeds this. The
    /// infeasibility path reuses it in `buildFarkasCert` to drop near-zero
    /// components from the Farkas witness — a sparsity cutoff with no
    /// gap-floor role, so the DESIGN RULE below doesn't bear on that use.
    ///
    /// Was 1e-6, which mistook genuine small-weight *binding* constraints
    /// for inactive ones. On near-circular DGGS hexagons (e.g. H3 r7–r10)
    /// the D-optimal design is degenerate: alternating vertices sit on the
    /// enclosing ellipse with true dual weight ~1e-7. At 1e-6 Newton polish
    /// zeroed those points, so the dual certificate under-counted `Z` and
    /// the duality gap floored at ~1.7e-6 — never reaching the strict 1e-6
    /// default no matter how many outer iterations ran (the iterate then
    /// oscillated as FW re-grew the weights and Newton re-zeroed them).
    /// At 1e-12 the binding constraints are retained, those cells converge
    /// to gap ~1.5e-7, and the genuine f64 gap floors at the finest
    /// resolution (S2 L30 / A5 r30, κ-driven, no sub-1e-6 weights to drop)
    /// are unaffected — they still DNC at 1e-6 as documented. 1e-12 sits
    /// above f64 roundoff dust (~1e-15) yet well below real small weights,
    /// matching the `tol.PSD_NEG_REL` noise-vs-signal scale. Still distinct
    /// from (and tighter than) the FW inner loops' `tol.WEIGHT_ACTIVE`.
    ///
    /// DESIGN RULE — keep this ≪ any `gap_tol` you intend to certify.
    /// Dropping a binding constraint of dual mass `m` inflates the gap by
    /// O(m), so the cutoff induces a gap floor of O(ACTIVE_THRESH)
    /// (measured: floor ≈ 1–2× the dropped weight). The original bug was
    /// the scale collision ACTIVE_THRESH == the default gap_tol == 1e-6:
    /// the certificate was as coarse as the bound it certified. At 1e-12
    /// the floor (~1e-12) is six orders below the 1e-6 default, so the
    /// discrete in/out switch is numerically invisible at that tolerance —
    /// which is why a *fixed low* value (vs a relative or adaptive one)
    /// suffices: the support is identified by optimality, and the residual
    /// switching noise is below what any practical gap_tol resolves.
    pub const ACTIVE_THRESH: f64 = 1e-12;

    /// Feasibility-cone margin for the backtracking b-update. Each
    /// outer step requires `min_i(b_new · xᵢ) ≥ FEAS_MARGIN`; α is
    /// halved up to MAX_BACKTRACKS times until the new b satisfies it.
    pub const FEAS_MARGIN: f64 = 1e-8;
    pub const MAX_BACKTRACKS: u32 = 30;

    /// Quasi-Newton b-update gate: only precondition the axis step by
    /// M⁻¹ when cond(M) exceeds this. For near-isotropic M (hex, DGGS
    /// cells, rotations near coordinate axes) the preconditioner adds
    /// sub-ULP direction noise that interacts badly with damping after
    /// Newton polish; the plain gradient step is used instead.
    pub const PRECOND_COND_MIN: f64 = 1.2;

    /// Skip the quasi-Newton machinery for the first `AXIS_WARMUP`
    /// outer iterations. Easy cases (hex, most DGGS cells) converge
    /// in ≤ this, so they pay zero preconditioner overhead.
    pub const AXIS_WARMUP: u32 = 2;

    /// Inner Frank-Wolfe budget for the MVEE weight solve, gated on input
    /// size. The outer loop interleaves ONE inner FW step with a Newton
    /// polish and a b-axis update per cycle (see the outer loop in
    /// `skar.zig`). For small point sets that schedule is optimal: Newton
    /// does the weight refinement and the whole solve finishes in 1–2 outer
    /// iterations, so extra inner FW steps would be pure overhead.
    ///
    /// But Newton's fraction-to-boundary step can never zero a weight, so it
    /// cannot shrink the *active set*; only an inner FW drop-step can. On
    /// inputs with many redundant / near-cocircular boundary points (the
    /// motivating case: A5 resolution-0 cells, whose default 320-point
    /// `cell_to_boundary` polygon has an enclosing ellipse touching only ~5
    /// corners) the active set then drains ~2 points per outer iteration, so
    /// the outer-iteration count scales with the point count and overruns
    /// `max_outer`. See `docs/a5_res0_dnc_report.md` for the full diagnosis.
    ///
    /// Fix: for inputs with more than `INNER_FW_BOOST_MIN_POINTS` working
    /// points, give the inner FW a real budget (`INNER_FW_BOOST_ITERS` steps,
    /// stopping early once the inner gap is below `INNER_FW_BOOST_TOL`) so it
    /// drains the active set within the first outer iteration; the subsequent
    /// Newton polishes then run on the true ~5-point support. This makes the
    /// outer count input-size-independent (A5 res-0: ~145 → ~6 outer iters,
    /// ≈500× faster) while leaving small inputs on the *bit-identical* 1-step
    /// path. A blanket boost was measured and rejected: it slows near-circular
    /// hexagons ~1.5× and worsens the genuine f64-floor finest-resolution
    /// cells, so the size gate is load-bearing, not cosmetic.
    ///
    /// The threshold sits above the largest "small cell" vertex count (DGGS
    /// cells are 4–10 points; cf. `n_hull = 10`) and far below a dense
    /// boundary (100s). Inputs above it have active-set draining to do, where
    /// the boost helps or is neutral; inputs below it are already near-minimal,
    /// where it is overhead.
    ///
    /// FUTURE: a fully-corrective / away-step inner FW that can drive weights
    /// to exactly zero would drain the active set without a large per-cycle
    /// budget, unifying the two regimes and removing this branch. Not yet
    /// attempted — this gate documents the regime split in the meantime.
    pub const INNER_FW_BOOST_MIN_POINTS: usize = 16;
    pub const INNER_FW_BOOST_ITERS: u32 = 100;
    pub const INNER_FW_BOOST_TOL: f64 = 1e-9;
};

/// Numerical tolerances — the "how small is small" guards.
/// These guard against divide-by-zero, underflow, and spurious convergence.
/// Tuned to f64 precision; not exposed to callers.
pub const tol = struct {
    /// Newton polish inner loop: stop when max-min of gradient components < this.
    pub const NEWTON_INNER: f64 = 1e-14;
    /// Newton polish: fraction-to-boundary step-size floor; below, declare stuck.
    pub const NEWTON_STEP_MIN: f64 = 1e-12;
    /// Hard floor for SolveError.NegativeDualityGap (FP noise below, bug above).
    pub const NEG_GAP: f64 = 1e-10;
    /// FW inner loops: minimum w_i to participate in the pairwise-swap candidate set.
    /// Distinct from (and looser than) algo.ACTIVE_THRESH, which is the *cert* cutoff.
    pub const WEIGHT_ACTIVE: f64 = 1e-14;
    /// Tiny-magnitude zero guard for norms and dot-products (`< tol ⇒ treat as 0`).
    pub const TINY: f64 = 1e-30;
    /// 2D det / scalar singular guard (denominator-is-zero cutoff).
    pub const NEAR_SING: f64 = 1e-15;
    /// halfspaceCheck: z.dot(z) ceiling below which FW cannot make progress.
    pub const FW_Z_EXHAUSTED: f64 = 1e-12;
    /// Underflow floor: pivot / scale / log argument.
    pub const UNDERFLOW: f64 = 1e-300;
    /// Relative cutoff for "FP noise" vs. "theorem violation" on values
    /// that should be ≥ 0 by PSD invariant (eigenvalues of A_perp,
    /// det of Minv). Below the threshold ⇒ silent clip; above ⇒ loud
    /// SolveError. Mirrors NEG_GAP's role for the gap.
    pub const PSD_NEG_REL: f64 = 1e-12;
};
