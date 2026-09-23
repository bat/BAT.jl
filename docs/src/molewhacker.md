# Molewhacker importance sampling

[`MolewhackerSampling`](@ref) builds a Gaussian mixture with local Fisher geometry
and draws fresh importance samples. The sampler is standalone within BAT.
Its API is experimental.

```julia
using BAT, Distributions, StableRNGs
using MeasureBase: Likelihood
import ForwardDiff, OptimizationLBFGSB

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

## Proposal construction

The proposal follows the [Newtrinos Molewhacker algorithm](https://github.com/Newtrinos-org/Newtrinos.jl/blob/bat-v5-migration/src/analysis/molewhacker.jl),
with Laplace seeds and fresh rounds added by default:

1. Transform the prior to standard-normal coordinates. By default, run parallel
   L-BFGS-B searches from ten Sobol starts to find initial centers. With Laplace
   seeds, each search stops after 50 iterations and a Newton step finishes it.
2. At each center, form a Gaussian with precision `I + J' F J`. Here `J` is the
   forward model's parameter Jacobian, and `F` is its observation distribution's
   Fisher information. The identity adds prior information once. Laplace seeds
   add a second Gaussian with the observed information as precision.
3. Let `q_uniform` be the equally weighted mixture of all local Gaussians.
   Assign component masses proportional to `target(center) / q_uniform(center)`.
4. Draw an initial discovery pool. Rank its points by their current
   `logtarget - logproposal` values. Add a Fisher Gaussian at each selected point.
   These added centers need not be modes.
5. Recompute all component masses using the center-ratio rule. From each new
   component, draw `floor(component_mass * previous_pool_size)` discovery points.
   Repeat until a hard budget or an explicit pool-ESS or efficiency threshold applies.
6. Run three more rounds by default, each after adding fresh draws from the
   current proposal to the pool. This step is an addition to the source algorithm.
7. Refit the proposal to those fresh draws by importance-weighted EM, keeping the
   adaptive mixture as a defensive share. This step is also an addition.

The adaptive pool guides proposal construction only. Its points have different
sampling laws, so reweighting the pool by the latest proposal does not produce
valid final importance weights. Its reported `pilot_ess` is a fitting heuristic.

## Final sampling law

By default, the final proposal is the fitted mixture. An explicit positive
`exploration_mass` adds a prior component after fitting:
`q_final = exploration_mass * prior + (1 - exploration_mass) * q_fitted`.
With bounded likelihood and positive prior mass, this bounds importance ratios. It does not guarantee
mode discovery. The prior component enters after fitting, preserving the source
algorithm's discovery proposals.

Freeze both the proposal and production count before drawing the final samples.
Conditional on all earlier work, these samples are IID from `q_final`.
Only these fresh samples enter the returned empirical measure. Their log
importance ratios are `logtarget - logproposal`.

Weights use one common exponential scale for numerical stability.
`diagnostics.logweight_scale` retains this scale. The target mass stays unchanged.
The normalized proposal appears in `result.approx`. Self-normalized estimates
retain their usual finite-sample bias.

## Controls and limits

- `nseeds` sets the number of initial centers. The default `init = nothing` uses
  Sobol starts in normal coordinates. `init = ExplicitInit(...)` supplies centers
  in original coordinates, which BAT copies and transforms.
- `init_mode` sets the initial optimizer. The default requires
  `import OptimizationLBFGSB`. It stops after 50 iterations with Laplace seeds,
  and after 1,000 without them. Set `init_mode = nothing` to keep supplied centers,
  or `nseeds = 0` to skip initialization. Prior-only discovery can fail badly on
  concentrated targets in higher dimensions.
- Mode searches receive equal shares of the available fitting-call budget, with
  remainder calls assigned in seed order. Each search owns its optimizer copy
  and RNG. A search that exhausts its share supplies no Gaussian. Unused calls
  remain available for adaptation.
- `ncandidates` sets the number of centers selected per round. It defaults to 14,
  independent of the thread count, so results do not depend on the machine.
  Geometry calculations use the selected `executor`.
- `executor = BAT.MultiThreadedExec(ntasks = 14)` limits concurrent BAT work to
  fourteen tasks, including target evaluation, geometry, and mixture scoring.
  The default `MultiThreadedExec()` uses `Threads.nthreads()` tasks. This setting
  does not change `ncandidates`, the Julia thread pool, or BLAS threads.
  Allocation-heavy likelihoods can run faster with fewer active tasks.
- `batchsize` sets the initial discovery-pool size and the independent sizing-pilot
  size. Later discovery batches follow component masses, so they can be empty.
  Target values at existing pool points are reused.
  Gaussians are cached by pool index, preserving every selection's mass update.
  Center density sums add only the selected occurrences each round.
  Mixture scoring uses the selected executor and bounded, reusable workspaces.
- `maxiter` is a strict limit on adaptation rounds. `maxiter = 0` skips discovery,
  adaptation, and fresh rounds. `maxcomponents` limits proposed occurrences,
  including repeated selections and any added prior component.
- `fresh_rounds` defaults to three. After adaptation stops, for any reason, each
  fresh round adds `batchsize` fresh draws from the current proposal to the pool,
  then selects and adds candidates as before. Discovery batches come only from
  new components, so the pool holds few draws from the current proposal. It then
  misses the ratio spikes that production draws hit, and one weight can dominate
  the output. Each fresh round costs `batchsize` target calls. Fresh rounds stop
  early at `maxevals`, `maxcomponents`, or when no candidate has finite weight.
  On the public DeepCore model, three fresh rounds cut the largest normalized
  squared weight from 0.15–0.58 to 0.05–0.11 over four seeds. On well-fitted
  targets they cost calls and gain nothing. Set `fresh_rounds = 0` for the source
  algorithm's rounds only.
- The fresh-round draws then refit the proposal, in the style of MitISEM
  (Hoogerheide, Opschoor and van Dijk). Each draw is weighted by the proposal that
  drew it, and importance-weighted EM fits one to six Gaussians to the target. The
  number maximizes the weighted log-likelihood of held-out draws, the cross-entropy part
  of KL(p || fit), and the refit on all draws starts from the held-out winner. The final
  proposal gives these 80% of the mass and keeps the adaptive mixture at 20% for
  defence. Center ratios see the target only at component centers, so they cannot
  see proposal mass placed where the target is small. The fit needs no extra target
  calls and is skipped when the fresh draws have too few effective samples. On six
  12-dimensional test targets it raised production ESS by 20–229%, and on the public
  DeepCore model by 19–36% over two seeds. Production draws come after the fit, from
  the frozen result, so they stay IID from one proposal. Pass a `MolewhackerRefit` as
  `refit` to tune the fit, or `nothing` to keep the adaptive mixture.
- `exploration_mass` defaults to zero. A positive value mixes the prior into the
  final proposal. This option leaves discovery unchanged.
- `laplace_seeds` defaults to `true`. It adds a second Gaussian at each seed, with
  the observed information `-∇² logtarget` as precision, variance inflated by
  `laplace_inflation` (default 1.2).
  Fisher information misses curvature where the forward model is stationary in a
  parameter, for example a mixing angle near maximal mixing. The Hessian comes
  from central differences of AD gradients, once per distinct seed. Where it is
  positive definite, one Newton step polishes the seed if the target increases.
  This step lets the default mode search stop early. Elsewhere, such as at kinks,
  the seed keeps its Fisher Gaussian alone. The two Gaussians share a center, so
  center-ratio fitting gives them equal mass. The target must support AD
  gradients, as for the default mode search. Laplace Gaussians count toward
  `maxcomponents`, so it must allow twice `nseeds`. Gradient calls for these
  Hessians do not count toward `maxevals`. Set `laplace_seeds = false` for the
  source algorithm's Fisher-only seeds.
- `maxevals` caps target calls, including mode searches, initial centers,
  discovery, sizing, and production. Geometry calls are separate and counted
  in `ngeometries`. The default adds no call limit beyond the other stopping
  rules. Explicit caps reserve production and any sizing pilot, and must leave
  room for enabled initialization and discovery. With automatic output, each new
  discovery draw also reserves one fresh output draw.
- `nsamples = nothing` is the default. Without an ESS goal, the output count
  matches the final discovery pool, or `batchsize` when adaptation is disabled.
  An explicit integer fixes the output count.
- `target_pool_ess` stops adaptation when the recycled-pool ESS exceeds its value.
  `target_efficiency` stops adaptation when that ESS divided by the pool size
  exceeds its value. Both default to `Inf`, leaving adaptation to the hard budgets.
  When both are set, either threshold can stop adaptation.
- A finite `target_ess` sizes production only. An independent fresh pilot estimates the output count.
  An explicit `nsamples` caps that count. With `nsamples = nothing`, only the
  remaining `maxevals` budget caps it. The achieved ESS is not guaranteed.
  Production never stops based on its current weights.

Choose adaptation thresholds separately from the production goal:

| Adaptation rule | Setting |
| --- | --- |
| Hard budgets only (default) | Leave both thresholds at `Inf` |
| Smaller fixed refinement budget | `maxiter = 16, ncandidates = 14` |
| Source pool-ESS heuristic | `target_pool_ess = 5000` |
| Source efficiency heuristic | `target_efficiency = 0.2` |
| Projected production ESS | `target_efficiency = target_ess / nsamples` |

A smaller `maxiter` trades refinement for a smaller mixture, independently of
the production sample count. With ten initial components, fourteen candidates,
and sixteen completed rounds, the mixture has 234 proposed occurrences. Laplace
seeds can add up to ten more, and the default fresh rounds up to 42. Reselection
can store fewer Gaussians. Failed geometries or earlier stops can reduce both
counts. Optional prior mixing can add one component.
This is a user-selected budget, not an automatic convergence test. Compare fresh
weighted estimates of the observables you need before reducing the budget.

The projected rule requires an explicit output cap and a goal below that cap.
It extrapolates from recycled-pool efficiency and can stop before finding tails or modes.
These thresholds do not validate the proposal or guarantee the production ESS.
Earlier versions coupled `target_ess` to pool-ESS stopping. Set
`target_pool_ess = target_ess` explicitly to retain that behavior.

Target draws follow the context RNG's serial order before parallel evaluation.
For deterministic optimizers without wall-time limits, changing the executor
preserves that order. The model must support the context's AD selector.

The sampler supports dense CPU geometry for Normal, MvNormal, Poisson,
Exponential, and product observation models. Singular local geometry rejects
that candidate. If initialization supplies no usable proposal, sampling uses
the prior. Unsupported models and unrelated errors propagate. An arbitrary
log-density closure does not expose the required forward model.

Product models share one parameter Jacobian. Diagonal and isotropic Normal
covariances use compact parameter charts. Whitened Jacobian rows form one Fisher
pullback. The default ForwardDiff path avoids a redundant primal model call.
Local Gaussians retain their precision factor, avoiding explicit inversion.

Dense parameter geometry needs quadratic storage and cubic factorization work.
Each round can propose `ncandidates` components, as in the source algorithm.
Reselecting a discovery point reuses its stored Gaussian. Its multiplicity remains
in the fitting density, mixture masses, and per-occurrence draw allocation.
Combining identical mixture categories can change draws for a fixed RNG seed
while preserving the proposal distribution.
With ten seeds and 14 candidates, 1,000 rounds can propose 14,010 components while
storing fewer Gaussians. `maxcomponents` counts proposed occurrences, preserving
the refinement budget even when components are reused.
Raising `nsamples` alone does not enlarge the discovery pool.
Mixture evaluations still grow with component count. Include initialization, geometry,
adaptation, and production when comparing total cost. These heuristics do not
establish global coverage or a general convergence guarantee.

## Diagnostics

`evalinfo.result` records iteration, target-call, geometry, component, and output
counts. `ncomponents` counts stored Gaussians. `ncomponent_proposals` includes
repeated selections and any added prior component. Geometry counts exclude cache
hits. `nseed_evals` counts initialization
target calls. `nseed_exhausted` counts mode searches that reach their assigned
budget. `nhessians` counts seed Hessians computed for `laplace_seeds`.
`niterations` counts adaptation rounds and `nfresh` counts fresh rounds. `history`
records pool growth, both component counts, and whether the round was fresh.
`stop_reason` describes adaptation, not the fresh rounds that follow.

`stop_reason` distinguishes `:maxiter`, `:maxcomponents`, `:maxevals`,
`:pilot_ess`, `:pool_efficiency`, `:no_finite_candidate`, and `:geometry_failure`.
`pilot_ess` describes the recycled discovery pool. Its efficiency is `pilot_ess / npilot`.
`pilot_efficiency` comes from
the independent sizing pilot, when enabled. `ess` and `efficiency` describe the
fresh production weights.

ESS averages weight dispersion and can hide one dominant weight. `pareto_k` is the
generalized Pareto shape of the largest production weights, as in PSIS. Values
above 0.7 mark unreliable estimates, and values below 0.5 are good. It is `NaN`
for fewer than 21 finite weights, and `Inf` when the largest weights exceed the
rest beyond the floating-point range. A small `pareto_k` does not rule out one
dominant weight, so also check `max_weight`, the largest normalized weight. Its
inverse is the L∞ effective sample size of Martino, Elvira and Louzada (2017).
Delta-method ESS values for single observables fail in the same case: a dominant
draw sits at the weighted mean and hides its own variance. Check relevant
observables and repeat runs when missed modes matter. Zero-target draws receive zero weight. A production
batch with no finite positive target mass fails.

`smooth_weights = true` replaces the largest production weights by expected
order statistics of that fit, capped at the largest raw weight (Pareto-smoothed
importance sampling). This lowers estimator variance and adds a small bias,
including for the target mass. `pareto_k` still describes the raw weights.
Smoothing cannot repair `pareto_k` above 0.7.

`weight_diagnostic` accepts an optional function of final log importance ratios.
For example, callers using MGVI can pass `MGVI.pareto_diagnostic`. Its return
value appears in `diagnostics.diagnostic`. It cannot replace the sampler's raw
weights. Its own sample-size requirements still apply.
