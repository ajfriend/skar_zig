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

    /// Certificate active-set cutoff: weights below this are dropped
    /// from `Info.cert`. Distinct from (and tighter than) the FW inner
    /// loops' `tol.WEIGHT_ACTIVE`.
    pub const ACTIVE_THRESH: f64 = 1e-6;

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
