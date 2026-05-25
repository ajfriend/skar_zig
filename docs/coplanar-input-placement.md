# Open question: should `coplanar_input` be an `Outcome` variant or an `InputError`?

Follow-up to [outcome-union-plan.md](outcome-union-plan.md). The plan
lists `coplanar_input: void` as the fourth `Outcome` variant. This
doc argues that placement is worth pressure-testing before the
refactor lands, because it sits awkwardly between an outcome and an
error.

## The asymmetry

Compare the three "input doesn't admit a cone" cases:

```zig
InputError.InsufficientPoints  // <3 points        → error
InputError.InvalidTolerance    // bad opts         → error
Outcome.coplanar_input         // rank-deficient X → variant
```

All three are properties of the *inputs* that disqualify them from
producing a cone. The first two are errors; the third is a variant.
The distinction isn't principled — it's "we happen to detect
coplanarity later in the pipeline." That's an implementation detail
bleeding into the type.

## The two questions that decide it

**1. When is coplanarity detected?**

- *Pre-flight check on `X` (e.g., rank of moment matrix before any
  iterations)* → it's input validation, same category as
  `InsufficientPoints`. Belongs in `InputError`.
- *Discovered mid-solve (e.g., singular Hessian at iterate)* → it's
  a legitimate algorithmic outcome. Belongs in `Outcome`.

If the answer is "both — we have an upfront check *and* a fallback
mid-solve detection," prefer error placement: the upfront check is
the common path, and the mid-solve case can map to the same error.

**2. What does a caller do differently with `.coplanar_input` vs.
`InputError.Coplanar`?**

If the answer is "same thing — log and bail," errors are strictly
lighter weight: no allocation, no switch arm, blanket `try`
propagates cleanly. If callers *want* to distinguish coplanarity
from other input failures (different recovery path, different
metric, different log line), that's an argument for keeping it
visible — but it's visible either way; an error variant is just as
matchable as a union variant.

## The case for moving it to `InputError`

- **Symmetry**: aligns with `InsufficientPoints`. Both are
  "structural problem with `X`, algorithm never engages."
- **Sharper `Outcome` invariant**: with coplanar gone, `Outcome`
  reads as "the solver ran and here's what it concluded" — three
  variants, each one earned by the algorithm doing work
  (`Converged` proved optimality, `Infeasible` proved infeasibility,
  `did_not_converge` ran out of budget). That's a cleaner story
  than "the solver concluded one of four things, including the case
  where we refused to start."
- **`void` payload is a smell**: a variant that carries no data is
  almost always either (a) an error in disguise or (b) waiting for
  data to be added later. Here it's (a).

## The case for keeping it in `Outcome`

- **Forces explicit acknowledgment**: a switch arm is harder to
  ignore than a `try` that swallows it. The whole point of the
  refactor is to make callers handle every outcome — moving
  coplanarity to errors *weakens* that for this case.
- **Consistency-within-the-union**: if everything algorithmically
  determinable about `X`'s structure lives in `Outcome`, the union
  is the complete story. Splitting it across two type categories
  fragments the mental model.
- **Future expansion**: if we later want to attach data (e.g., the
  great-circle normal as a witness to coplanarity), a variant
  trivially gains a payload; an error variant would have to migrate
  back to the union.

## Recommendation

Lean toward **moving `coplanar_input` to `InputError`**, contingent
on the answer to question 1. The symmetry with `InsufficientPoints`
is hard to argue against, and a `void` variant is a weak use of the
union machinery. If detection happens both upfront and mid-solve,
the upfront path is the dominant one and the algorithmic-outcome
framing doesn't survive scrutiny.

If we decide to keep it as a variant, the justification should be
spelled out in the plan doc — specifically, that we *want* to force
callers to acknowledge it via the switch rather than let it
propagate through `try`. That's a defensible choice, but it should
be the stated reason, not a default.

## Scope

This is a one-line decision; not a re-architecting. Either:

- Delete `coplanar_input` from `Outcome`, add
  `InputError.CoplanarInput`, return it from `solve` before
  constructing any variant. ~5 lines changed in `src/skar.zig`,
  one test arm migrated, one example arm dropped.
- Or keep the plan as-is and add a sentence to the plan doc
  defending the variant placement.

Decide before the refactor lands so the migration moves through one
shape, not two.
