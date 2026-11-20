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
- Moments of measures (`mean`, `var`, `cov`) come from MeasureBase, BAT
  keeps `median` and `mode`.
- `truncate_batmeasure` truncates measures with scalar variates via
  MeasureBase's `truncate`, without renormalizing, and keeps BAT's own
  truncation for products of univariate distributions.

## Variate shapes and ValueShapes

`NamedTupleDist` is replaced by MeasureBase products over named tuples.
Shapes are derived from measures by ValueShapes' MeasureBase extension
(branch `measurebase` of ValueShapes: `varshape`, `unshaped`, shapes
applied to measures, `resultshape` of transports, and `asmeasure` for
`NamedTupleDist`, `ConstValueDist` and `ReshapedDist`).
`DensitySampleVector` stores the variates in the struct-of-arrays layout
that MeasureBase's `rand` produces (flat arrays only for flat variate
spaces), with constants as `Fill` columns, and derives `varshape` from
it. ValueShapes remains in use per variate: parameter names and
selections, flat conversions and the BAT 5.x HDF5 layout.

## Conventions

MeasureBase's conventions apply: densities are `-Inf` outside the
support, transports `NaN` outside the support of the source, wrong shapes
throw. BAT's boundary tweaks (clamping uniform inputs, `-1e38`
substitutes) are not ported into MeasureBase; the ones adopted there are
listed in MeasureBase's notes.

## Random streams

MCMC derives its random streams with the stream keys of
HeterogeneousComputing (`rootrng`, `subrng`). A run starts from a fresh
keyed root: a child of the context RNG under a drawn nonce if that RNG has
a stream key, so the run keeps its kind of generator, otherwise a root
whose entropy is drawn from the context RNG. Chains are children
`(:chain, id)` of the run root, and each step draws from one child
`(:step, cycle, stepno)` of its chain. The selection of the active
proposal draws from the step RNG first. Proposals that declare batched
draws (random walk, MALA, global proposal) then draw the variates of all
walkers in one call and the acceptance uniforms in a second one, both in
the order of the walker ids. All other proposals, like HMC, get one child
of the step RNG per walker. Tuning has its own namespace: an HMC proposal
derives a tuning stream root `(:tuning)` from the RNG it is created with,
which is the chain RNG or, for a component of a multi-proposal, its child
`(:proposal, i)`. The step-size probes derive one child per probing event
and walker from that root.

The algorithm of the default generator is not part of the API, since
PureRNGs.jl is to be supported later. Host loops and device fills only
give the same numbers once both use a common counter-based generator.

## Accelerators

The MH sampler with RAM tuning is the first candidate. Known obstacles,
in order: `findall` plus scatter in the accept step,
`PositiveFactorizations` and rank-one Cholesky updates in the tuner,
`checked_logdensityof`'s `try`/`catch` in the hot loop, and no
compute-unit plumbing into the tuner states. Device chain states must
take their root from an RNG on the compute unit of the context.
