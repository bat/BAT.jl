# Molewhacker importance sampling

[`MolewhackerSampling`](@ref) fits a defensive Gaussian mixture and draws fresh
importance samples. It is a standalone BAT sampler. It does not require MGVI.
The API is experimental.

```julia
using BAT, Distributions, StableRNGs
using MeasureBase: Likelihood
import ForwardDiff

posterior = PosteriorMeasure(
    Likelihood(z -> Normal(z[1], 0.5), 2.0),
    MvNormal([0.0], [1.0;;]),
)
context = BATContext(rng = StableRNG(71), ad = ForwardDiff)
algorithm = MolewhackerSampling(nsamples = 4000, batchsize = 512, maxiter = 5)
result = evalmeasure(posterior, algorithm, context)
samples = BAT.samplesof(result)
diagnostics = result.evalinfo.result
```

This posterior has mean `1.6` and variance `0.2`. `bat_sample` also accepts the
algorithm. Use `evalmeasure` to retain the fitted proposal and diagnostics.

## Sampling law

BAT first transforms the prior to standard-normal coordinates. At each candidate
center, the sampler forms the precision `I + J' F J`, using the forward model's
Jacobian and the observation distribution's Fisher information. The Gaussian
uses that center as its mean. It need not be a mode.

Each training draw retains its actual generating log density. The sampler ranks
candidate centers by the current target-to-proposal ratio. It fits each new
component's mass and variance scale against an estimate of
`J(q) = integral(target(z)^2 / q(z))`.

For a new Gaussian `g`, the candidate proposal is
`(1 - beta) * q + beta * (epsilon * prior + (1 - epsilon) * g)`.
A bounded scalar search fits `beta`. `covariance_scales` supplies the finite set
of variance multipliers. A fresh draw from the equal mixture of the old and
candidate proposals validates the best fitted change. Validation uses its own
generating density in the second-moment estimate.

After adaptation, the sampler freezes the proposal and production count.
Conditional on all earlier work, the final draws are IID from that proposal.
Only those draws enter the returned empirical measure. Their log importance
ratios equal `logtarget - logproposal`.

Weights use one common exponential scale for numerical stability.
`diagnostics.logweight_scale` retains this scale. The sampler leaves the target
mass unchanged. The fitted proposal is normalized and stored in `result.approx`.
Self-normalized estimates retain their usual finite-sample bias.

## Tuning and limits

- `exploration_mass` keeps a positive prior coefficient through every update.
  With bounded likelihood, this bounds the importance ratios. It does not prove
  that the sampler discovers every mode.
- `batchsize` controls training, validation, and sizing batches. The sampler
  reuses training target values and records their generating densities.
- `maxiter`, `maxcomponents`, and `maxevals` bound adaptation. `maxiter = 0`
  draws from the initial proposal without training.
- `nsamples` reserves the production budget. `maxevals` also counts training,
  validation, sizing, center-refinement, and mode-search target calls. Fisher model/Jacobian calls
  are separate and counted as geometry attempts in `ngeometries`.
- A finite `target_ess` enables a fresh sizing pilot. The pilot chooses a count
  between one and `nsamples` before production starts. It cannot guarantee the
  achieved ESS. Production never stops based on its current weights.
- `nseeds` and `init = ExplicitInit(...)` supply initial centers in user
  coordinates. The sampler copies and transforms them. Each valid seed starts
  with equal mass within the non-prior part of the proposal. Exact duplicate
  centers share one Gaussian and retain their combined seed mass.
- `mode = OptimAlg(...)` optionally refines initial and discovered centers.
  Load the chosen optimizer backend. The search maximizes the transformed
  target density and shares the target-call budget.
- `refine_centers = true` also fits one Fisher-gradient step from each discovered
  center when `mode` is `nothing`. The original center remains a candidate.
  This requires gradients of the transformed target through the context's AD
  selector. It uses the original center's precision and caps the step at
  `sqrt(d)` in that metric. It does not move explicit initial seeds.
  The extra target calls and candidate fits can help poorly centered proposals,
  especially in higher dimensions. They can also reduce the number of fitting
  rounds under a tight budget. The default is `false` because observable errors
  and total cost can worsen even when production ESS improves.
- `ncandidates` limits candidate attempts independently of the executor.
  Discovery excludes centers inside a previous candidate's unit Fisher
  ellipsoid, so narrow nearby features can remain distinct.
  Target draws follow the context RNG's serial order before parallel density
  evaluation. The model must support the supplied AD selector.
- `patience` counts consecutive unaccepted rounds, including training rounds
  with no positive target mass. Every completed round appears in history.

The initial implementation supports dense CPU geometry for Normal, MvNormal,
Poisson, Exponential, and product observation models. Singular local geometry
rejects that candidate and preserves the last valid proposal. Unsupported models
and unrelated errors propagate. Arbitrary log-density closures do not provide
the required forward model.

The defaults are bounded heuristics. Large observation covariance derivatives
and growing training archives can be expensive. Dense parameter geometry costs
quadratic storage and cubic factorization work. No setting establishes global
coverage or a general convergence guarantee.

Product models share one parameter Jacobian across their factors. This avoids
repeated differentiation of the complete model, but stores all factor-parameter
rows together. Its Jacobian storage scales with observation-parameter count
times target dimension.

Proposal densities use the distribution library's batched in-place API, with
scalar fallbacks for mixed-precision buffers and indeterminate tail values.
Local Gaussians retain the validated precision factor at unit covariance scale.
Other scales refactor the scaled matrix, preserving its numerical validity check.

Include fitting and geometry cost when comparing samplers. Higher production ESS
need not reduce total cost or error for a specific observable.

## Diagnostics

`evalinfo.result` records `stop_reason`, iteration and evaluation counts,
component and failed-geometry counts, final ESS and efficiency, sizing-pilot
efficiency, and validation history. Stop reasons distinguish iteration,
component, and evaluation limits from `:no_improvement`.

ESS measures weight concentration. Inspect estimates of relevant observables
and repeat runs when missed modes matter. Individual zero-target draws receive
zero weight. A production batch with no finite positive target mass fails.

`weight_diagnostic` accepts an optional function of the final log importance
ratios. For example, callers already using MGVI can pass
`weight_diagnostic = MGVI.pareto_diagnostic`. The returned value appears in
`diagnostics.diagnostic`. Diagnostic functions receive a copy and cannot replace
the sampler's raw weights. Their own sample-size requirements still apply.
