# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    ImportanceSamplersSampling(; sampler, pretransform=NormalBased(), threaded=true)

*Experimental feature, not part of stable public API.*

Sample with an ImportanceSamplers.jl algorithm. Requires ImportanceSamplers.jl
to be loaded. `sampler` supplies the proposal, sample count and adaptation rules.
Its proposal must use flat vectors in the coordinates selected by `pretransform`.
BAT applies coordinate transforms and their Jacobians exactly once.

This adapter uses CPU execution. `BATContext` supplies the RNG and AD backend.
Numerical precision follows the supplied proposal and target. With `threaded=true`,
sampling and result conversion use Julia's default thread pool. The target,
coordinate transforms and their data must support concurrent evaluation.

`evalmeasure` returns weighted BAT samples and a logarithmic mass estimate.
`evalinfo(result).result.samples` retains the complete native weighted result,
including log weights, proposal/round provenance and sampler diagnostics.
The adapter reevaluates the target once per returned sample to populate BAT's
log-density column. Native evaluation counts exclude that conversion pass.
No evidence uncertainty or MCMC effective sample size is inferred.

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct ImportanceSamplersSampling{S,TR<:TransformIntent} <:
                AbstractSamplingAlgorithm
    "ImportanceSamplers.jl algorithm, with proposals in transformed flat coordinates."
    sampler::S

    "BAT coordinate transformation."
    pretransform::TR = (pkgext(Val(:ImportanceSamplers)); NormalBased())

    "Evaluate independent samples on Julia's default thread pool."
    threaded::Bool = true
end
export ImportanceSamplersSampling
