# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

## Ensemble MCMC proposals

[`EnsembleProposal`](@ref) uses [EnsembleMCMC.jl](https://github.com/JuliaBayes/EnsembleMCMC.jl)
for Stretch, differential-evolution, snooker moves, and fixed move mixtures:

```julia
using BAT
import EnsembleMCMC

moves = EnsembleMCMC.MoveMixture(
    (EnsembleMCMC.StretchMove(), EnsembleMCMC.DEMove()), [4, 1]; schedule=:cycle)
algorithm = TransformedMCMC(proposal=EnsembleProposal(moves), nwalkers=32)
```

Load EnsembleMCMC explicitly. Set `nwalkers` explicitly and supply enough walkers
for the dimension and chosen moves. Initial transformed coordinates must be finite
and span their dimension. BAT retains its prior transforms and retry initialization.

Each BAT chain is one coupled ensemble. Each step completes one sweep over all
walkers. ESS uses the aligned ensemble-mean process and pools independent BAT
ensembles. Storing burn-in disables ensemble ESS because histories may not align.

The default executor is `EnsembleMCMC.SerialExecutor()`. Pass
`executor=EnsembleMCMC.ThreadedExecutor()` to `EnsembleProposal` for concurrent
walker evaluation within each group. The target must support concurrent calls.
Groups remain ordered, and deterministic targets retain seeded replay across threads.

For CPU batch evaluation, pass `batch_logdensity!`:

```julia
function batch_density!(values, transformed_target, positions)
    values .= BAT.checked_logdensityof.(Ref(transformed_target), eachcol(positions))
    return nothing
end
proposal = EnsembleProposal(moves; batch_logdensity! = batch_density!)
```

This scalar broadcast illustrates the contract. Replace it with a faster batch
calculation when available. The callback receives a transformed, unshaped target
and must include its full density, including prior and Jacobian terms. It fills
every output, treats positions as read-only, and retains neither borrowed array.
It controls its own parallelism. BAT shares cached initial log densities with
every ensemble component, avoiding repeated target calls during construction.
Cheap targets may run faster with the default scalar path.

An `EnsembleProposal` can also join a fixed-weight `MCMCMultiProposal`, including
ordinary BAT proposals. BAT reports cycle counts for each outer component and
cumulative counts for the moves inside each ensemble proposal.

Pass `record_transitions=true` to keep per-sweep `transitions` and `walker_ids`
in proposal diagnostics. Each owned record contains the BAT `cycle`, BAT `step`,
inner `move_index`, `accepted`, and `acceptance_probabilities`. These records
include rejected sweeps and initialization and burn-in cycles, even when their
samples are discarded. Filter by cycle for the phase of interest. Storage grows
with sweeps times walkers. The default retains only cumulative counts.
Sample proposal IDs identify outer BAT components. BAT owns RNG addresses for
scalar, threaded, and batched execution.

After `em = evalmeasure(target, algorithm, context)`, access direct ensemble
records through `evalinfo(em).result.chain_diagnostics[chain].transitions`.
For an outer mixture, use
`evalinfo(em).result.chain_diagnostics[chain].components[component].diagnostics.transitions`.

The adapter requires `RepetitionWeighting`, `NoAdaptiveTransform`, and
`NoMCMCTransformTuning`. Ensemble moves use `NoMCMCProposalTuning`.
Adaptive mixture-weight tuning is unsupported. See the package documentation for
move laws and walker requirements.

```@docs
ARPWeighting
BAT.batalgorithm
bat_compare
bat_integrated_autocorr_len
bat_marginalmode
BAT.auto_renormalize
BAT.BinnedModeEstimator
BAT.convert_for
BAT.DistributionTransform
BAT.LowRankAffineTransform
BAT.PathfinderTransformInit
BAT.enable_error_log
BAT.error_log
BAT.EvalException
BAT.evalmeasure_impl
BAT.ext_default
BAT.get_adselector
BAT.get_valid_adselector
BAT.PackageExtension
BAT.pkgext
BAT.set_rng
batmeasure
BridgeSampling
EllipsoidalNestedSampling
EllipticalSliceMCMCSampling
GridSampler
HierarchicalDistribution
PriorImportanceSampler
ReactiveNestedSampling
SobolSampler
EnsembleProposal
truncate_batmeasure
ValueAndThreshold

BAT.validate_evalmeasure
BAT.MCMCChainState
BAT.MCMCChainStateInfo
BAT.MCMCIterator
BAT.MCMCProposal
BAT.MCMCProposalState
BAT.MCMCProposalTunerState
BAT.MCMCState
BAT.MCMCTempering
BAT.MCMCTransformTunerState
BAT.MeasureEvalInfo
BAT.PolarShellDistribution
BAT.SimpleMCMCProposalState
BAT.TemperingState
```
