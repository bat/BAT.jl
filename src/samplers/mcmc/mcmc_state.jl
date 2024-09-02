# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# TODO: MD, adjust docstring to new typestructure
"""
    mutable struct MCMCState end

*BAT-internal, not part of stable public API.*

Represents the current state of an MCMC state.

!!! note

    The details of the `MCMCState` and `MCMCProposal` API (see below)
    currently do not form part of the stable API and are subject to change
    without deprecation.

To implement a new MCMC algorithm, subtypes of both [`MCMCProposal`](@ref)
and `MCMCState` are required.

The following methods must be defined for subtypes of `MCMCState` (e.g.
`SomeMCMCState<:MCMCState`):

```julia
BAT.get_proposal(state::SomeMCMCState)::MCMCProposal

BAT.mcmc_target(state::SomeMCMCState)::BATMeasure

BAT.get_context(state::SomeMCMCState)::BATContext

BAT.mcmc_info(state::SomeMCMCState)::MCMCStateInfo

BAT.nsteps(state::SomeMCMCState)::Int

BAT.nsamples(state::SomeMCMCState)::Int

BAT.current_sample(state::SomeMCMCState)::DensitySample

BAT.sample_type(state::SomeMCMCState)::Type{<:DensitySample}

BAT.samples_available(state::SomeMCMCState, nonzero_weights::Bool = false)::Bool

BAT.get_samples!(samples::DensitySampleVector, state::SomeMCMCState, nonzero_weights::Bool)::typeof(samples)

BAT.next_cycle!(state::SomeMCMCState)::SomeMCMCState

BAT.mcmc_step!(
    state::SomeMCMCState
    callback::Function,
)::nothing
```

The following methods are implemented by default:

```julia
getalgorithm(state::MCMCState)
mcmc_target(state::MCMCState)
DensitySampleVector(state::MCMCState)
mcmc_iterate!(state::MCMCState, ...)
mcmc_iterate!(states::AbstractVector{<:MCMCState}, ...)
isvalidstate(state::MCMCState)
isviablestate(state::MCMCState)
```
"""
mutable struct MCMCState{
    M<:BATMeasure,
    PR<:RNGPartition,
    FT<:Function,
    TP<:TransformedMCMCProposal,
    Q<:Distribution{Multivariate,Continuous},
    S<:DensitySample,
    SV<:DensitySampleVector{S},
    CTX<:BATContext
} <: MCMCIterator
    target::M
    proposal::TP
    f_transform::FT
    samples::SV
    sample_z::S
    info::MCMCStateInfo
    rngpart_cycle::PR
    nsamples::Int64
    stepno::Int64
    context::CTX
end

export MCMCState

function MCMCState(
    sampling::MCMCSampling,
    target::BATMeasure,
    id::Integer,
    v_init::AbstractVector{P},
    context::BATContext
    ) where {P<:Real}
    
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))
    
    n_dims = getdof(target)
    proposal = _get_proposal(sampling, target, context, v_init) # TODO: MD Resolve handling of algorithms as proposals 
    stepno::Int64 = 0

    cycle::Int32 = 1
    nsamples::Int64 = 0

    adaptive_transform_spec = _get_adaptive_transform(sampling) # TODO: MD Resolve
    g = init_adaptive_transform(adaptive_transform_spec, target, context)

    logd_x = logdensityof(target, v_init)
    inverse_g = inverse(g)
    z = inverse_g(v_init)
    logd_z = logdensityof(MeasureBase.pullback(g, target), z)

    W = Int # TODO: MD: Resolve weighting schemes in transformed MCMC
    T = typeof(logd_x)

    info = MCMCSampleID(Int32(id), cycle, 1, CURRENT_SAMPLE)
    sample_x = DensitySample(v_init, logd_x, one(W), info, nothing)

    samples = DensitySampleVector{Vector{P}, T, W, MCMCSampleID, Nothing}(undef, 0, n_dims)
    push!(samples, sample_x)

    sample_z = DensitySample(z, logd_z, one(W), info, nothing)

    state = MCMCState(
        target,
        proposal,
        g,
        samples,
        sample_z,
        MCMCStateInfo(id, cycle, false, false),
        rngpart_cycle,
        nsamples,
        stepno,
        context
    )

    # TODO: MD, is the below necessary/desired? 
    reset_rng_counters!(state)

    state
end

get_proposal(state::MCMCState) = state.proposal

mcmc_target(state::MCMCState) = state.target

get_context(state::MCMCState) = state.context

mcmc_info(state::MCMCState) = state.info

nsteps(state::MCMCState) = state.stepno

nsamples(state::MCMCState) = state.nsamples

current_sample(state::MCMCState) = current_sample(state, state.proposal)

sample_type(state::MCMCState) = eltype(state.samples)


