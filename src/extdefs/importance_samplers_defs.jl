# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    ImportanceSamplersSampling(; sampler, pretransform=NormalBased(), threaded=true)

*Experimental feature, not part of stable public API.*

Sample with an ImportanceSamplers.jl algorithm. Requires ImportanceSamplers.jl
to be loaded. `sampler` supplies the proposal, sample count and adaptation rules.
Its proposal must use flat vectors in the coordinates selected by `pretransform`.
BAT applies coordinate transforms and their Jacobians exactly once.

`BATContext` selects the compute device, RNG and AD backend. Numerical precision
follows the proposal and target. CPU execution uses Julia's default thread pool
with `threaded=true`. Device execution uses batched densities and transports and
returns device-resident samples and weights. Targets, transforms and proposals
must support the selected device, including the AD backend for gradient methods.
Data-bearing likelihoods need device adaptation
rules, as in other Julia GPU code. Opaque closures with captured arrays cannot
be transferred. Device execution requires `threaded=true`.

`evalmeasure` returns weighted BAT samples and a logarithmic mass estimate.
`evalinfo(result).result.samples` retains the complete native weighted result,
including log weights, proposal/round provenance and sampler diagnostics.
The adapter retains native target values for BAT's log-density column.
Result conversion does not evaluate the target again.
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
