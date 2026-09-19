# BAT on the new MeasureBase (branch `new-mb`)

Working notes for the migration of BAT to MeasureBase 0.15, for reviews
and for guiding the next steps. Kept up to date while the branch evolves,
to be removed before the merge.

## Goal

BAT stops carrying its own measure layer. MeasureBase's batched-first
measures, transports and random variates (see `redesign.md` in the
MeasureBase repository, branch `major-upgrade`) replace what BAT had
added to work around the performance and autodiff limits of the old
MeasureBase. Later, BAT's samplers keep arrays on accelerators, using the
`BATContext` compute unit that was prepared for this.

## What goes, what stays

- Gone: the `BATMeasure` supertype (BAT's measure types subtype
  `AbstractMeasure`), the distribution wrappers (`asmeasure`), the power,
  pushforward, weighted and superposition duplicates, BAT's standard
  uniform and normal distributions (`StdUniform`, `StdNormal` and their
  powers), `DistributionTransform` and the whole transform implementation
  (`transport_to`), `HierarchicalDistribution` (`mbind`).
- Deprecated: `lbqintegral` (`mintegrate_exp`) and `distbind` (`mbind`).
- Kept: `PosteriorMeasure` as a thin `AbstractMeasure` subtype
  (likelihood plus prior, zero-prior short-circuit), `EvaluatedMeasure`,
  `DensitySampleMeasure`, `BispacedMeasure`, the transform intents and
  `bat_transform`, `checked_logdensityof` as BAT's downstream check.
- `batmeasure` becomes a canonicalization: `asmeasure` for anything
  MeasureBase accepts, MeasureBase density measures (possibly nested)
  become posterior measures, named tuples of distributions become measure
  products. `distprod` keeps its meaning and returns MeasureBase products.

## Variate shapes and ValueShapes

`NamedTupleDist` is replaced by MeasureBase products over named tuples.
Shapes are derived from measures by ValueShapes' MeasureBase extension
(branch `measurebase` of ValueShapes: `varshape`, `unshaped`, shapes
applied to measures, `resultshape` of transports, and `asmeasure` for
`NamedTupleDist`, `ConstValueDist` and `ReshapedDist`). Sample storage
keeps flat vectors plus shapes for now. Decided: `DensitySampleVector`
moves to struct-of-arrays storage, the layout MeasureBase's `rand`
produces (flat arrays only for flat variate spaces), as a separate step,
which removes most of the remaining ValueShapes use.

## Conventions

MeasureBase's conventions apply: densities are `-Inf` outside the
support, transports `NaN` outside the support of the source, wrong shapes
throw. BAT's boundary tweaks (clamping uniform inputs, `-1e38`
substitutes) are not ported into MeasureBase; the ones adopted there are
listed in MeasureBase's notes.

## Accelerators

The MH sampler with RAM tuning is the first candidate. Known obstacles,
in order: per-walker mutable Philox RNGs re-seeded every step, `findall`
plus scatter in the accept step, `PositiveFactorizations` and rank-one
Cholesky updates in the tuner, `checked_logdensityof`'s `try`/`catch` in
the hot loop, and no compute-unit plumbing into the tuner states. The RNG
design needs a decision before this work starts.

## Open questions

- `truncate_batmeasure`: keep its renormalizing semantics or re-express
  it on `restrict` (which only masks).
- The RNG design for batched device sampling.
- Moments and modes of measures (`mean`, `var`, `cov`, `mode`) are
  defined by BAT for MeasureBase types; they belong in MeasureBase.
