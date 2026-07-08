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

    /// Sparse Frank-Wolfe weight initialization, gated on input size.
    ///
    /// The MVEE inner solve starts from a weight vector and lets FW move mass
    /// onto the support. FW *grows* the support well (each step adds the most
    /// violated point) but *prunes* it poorly: only a pairwise drop-step can
    /// remove a point, and Newton's fraction-to-boundary step can never zero a
    /// weight. So the uniform start `w_i = 1/nw` (every point active) is the
    /// worst case for any input whose support is a small subset — the whole
    /// solve becomes a slow drain. On A5 resolution-0 cells (the 320-point
    /// `cell_to_boundary` polygon whose enclosing ellipse touches only ~5
    /// corners) that drain is ~2 points/outer-iter, so the outer count scales
    /// with the point count and overruns `max_outer`. See
    /// `docs/a5_res0_dnc_report.md` for the full diagnosis.
    ///
    /// Fix: for inputs with more than `SEED_SPARSE_MIN_POINTS` working points,
    /// seed only `SEED_SPARSE_K` well-spread extreme points (greedy
    /// farthest-point; `farthestPointSeed` in `skar.zig`) so FW *grows* into the
    /// support instead of draining a full active set. This is the textbook MVEE
    /// initialization (Kumar–Yıldırım). Measured vs the uniform start: A5 res-0
    /// converges ~56× faster and genuine medium/large inputs ~3–6× faster, with
    /// the plain 1-step inner FW and unchanged aspect ratios.
    ///
    /// The size gate is load-bearing: on small *near-circular* cells (e.g. H3
    /// hexagons) the uniform start is already the symmetric optimum, and a
    /// sparse seed breaks that symmetry and *slows* them (~1× → ~11 outer iters).
    /// The threshold sits above the largest "small cell" vertex count (DGGS
    /// cells are 4–10 points; cf. `n_hull = 10`) and far below a dense boundary
    /// (100s). Below it, inputs keep the bit-identical uniform start.
    ///
    /// FUTURE: the true discriminator is "redundant / non-symmetric," not size —
    /// a cheap proxy could also accelerate small *irregular* polygons (which the
    /// size gate skips). And a fully-corrective / away-step inner FW that drives
    /// weights to exactly zero could remove the regime split entirely.
    pub const SEED_SPARSE_MIN_POINTS: usize = 16;
    pub const SEED_SPARSE_K: usize = 5;

};

/// Tuning for the EXPERIMENTAL trust solver path (`src/trust.zig`,
/// `SolveOptions.method = .trust`): trust-region descent on the reduced
/// convex objective h(b) over the sphere, with the alternating path's inner
/// MVEE machinery as the oracle. Prototype values — not yet tuned.
pub const trust = struct {
    /// Inner MVEE oracle per h-evaluation: FW in bursts of INNER_BURST
    /// steps with a stall exit — stop when a burst improves the design
    /// value by less than INNER_STALL_REL·(1+|h|) — up to the
    /// INNER_ITERS total budget, then ONE Newton polish. The stall exit
    /// is what keeps κ-limited cells (whose g-noise sits above any
    /// reachable INNER_TOL) from grinding the whole budget at noise
    /// amplitude; the burst FW itself remains monotone-in-intent with
    /// no snapshot/restore machinery, and the single final polish means
    /// the returned state is inner-(near-)optimal so the envelope
    /// gradient −3·c is the gradient of the h reported. History note:
    /// a rounds/burst/patience oracle with per-round polish and best-w
    /// tracking was tried and reverted — it could return under-refined
    /// snapshot states whose reported gradient wasn't the gradient of
    /// the reported h, which the trust region reads as a
    /// systematically wrong slope (measured ρ → −7.95 as Δ → 0 on
    /// cap82). Floor-regime cert pathologies are handled by the RECERT
    /// phase instead.
    pub const INNER_ITERS: u32 = 320;
    pub const INNER_BURST: u32 = 64;
    pub const INNER_TOL: f64 = 1e-11;
    pub const INNER_STALL_REL: f64 = 1e-9;
    /// Certify an accepted trust-region iterate only once the accepted
    /// step's predicted decrease has fallen to within a couple of
    /// orders of gap_tol: while the model still predicts ≫ gap_tol of
    /// remaining descent, no certificate can pass and computing one is
    /// pure overhead (early wide-cap iterates). pred is in the same
    /// units as the gap, so this gate is scale-aware — a ‖g‖-based gate
    /// was tried first and mis-fired on elongated regions whose Hessian
    /// scale ≫ B0 (states survey: certificates that would have passed
    /// were skipped). The iteration-0 certificate is always computed
    /// (it is what makes already-optimal inputs converge in 0
    /// iterations), and the RECERT phase always certifies.
    pub const CERT_PRED_FACTOR: f64 = 100.0;
    /// Trust-region radius: initial, max, shrink on rejection, growth
    /// on a very successful step (ratio ≥ ETA_GOOD with a full-length
    /// step), and the collapse threshold that ends the solve. Radii
    /// are in tangent-plane units (≈ tan of the axis rotation angle).
    pub const DELTA0: f64 = 0.5;
    pub const DELTA_MAX: f64 = 4.0;
    pub const SHRINK: f64 = 0.25;
    /// Gentler shrink for ACCEPTED-but-poor steps (ρ < RHO_POOR):
    /// progress was made, the radius just overshot the model's
    /// fidelity range — 0.25 here makes δ leapfrog the fidelity
    /// boundary and oscillate with GROW (measured on cap89: alternating
    /// ρ ≈ 0.2 / ρ ≈ 0.8 iterations).
    pub const SHRINK_POOR: f64 = 0.5;
    pub const GROW: f64 = 2.0;
    pub const DELTA_MIN: f64 = 1e-14;
    /// Step acceptance thresholds on ρ = actual/predicted decrease.
    /// ETA gates acceptance; RHO_POOR triggers a radius shrink even on
    /// an accepted step (the textbook ρ < ¼ rule — without it the loop
    /// can creep at ρ ≈ 0.15 forever when third-order terms of h
    /// dominate the quadratic model over the current radius, measured
    /// on cap89 under the majorant model: 83 iterations → 15 with the
    /// rule); ETA_GOOD + a radius-limited step triggers growth.
    pub const ETA: f64 = 0.05;
    pub const RHO_POOR: f64 = 0.25;
    pub const ETA_GOOD: f64 = 0.7;
    /// Fallback isotropic model Hessian B = B0·I, used when the
    /// per-evaluation majorant Hessian goes non-PD (roundoff or
    /// far-field states). 3·I is the majorant Hessian's own limit at a
    /// circular optimum — the fallback is the derived value, not a fit.
    pub const B0: f64 = 3.0;
    /// Exit the trust-region loop (to the RECERT phase) when the
    /// step's predicted decrease falls below the merit function's own
    /// resolution, pred ≤ PRED_NOISE_REL·(1+|h|): the ratio test can
    /// never verify such a step, so every trial is a rejection and Δ
    /// just marches to its floor one oracle evaluation at a time
    /// (measured on the H3 r9 CANARY cell: |g| = 3e-10, pred = 2e-20,
    /// 26 identical rejections before the re-cert phase fixed it in
    /// one attempt).
    pub const PRED_NOISE_REL: f64 = 1e-14;
    /// Re-certification attempts after the trust region stalls without
    /// a certified gap ≤ tol. Near the f64 gap floor the constructed
    /// certificate is sensitive to the incidental weight state at
    /// noise amplitude (measured on A5 res-30: the first cert's
    /// M-Cholesky fails for the alternating path too — it succeeds on its
    /// second outer iteration purely by re-sampling w). Each attempt
    /// re-runs the oracle at the fixed near-optimal axis (FW steps at
    /// noise level + a fresh polish perturb w) and re-certifies.
    /// Bounded so genuinely floored cells stop instead of burning the
    /// whole outer budget at oracle prices.
    pub const RECERT_MAX: u32 = 32;
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
