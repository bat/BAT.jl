# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct HamiltonianMC <: MCMCProposal

The [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
(HMC) sampling algorithm, using the multinomial no-U-turn sampler (NUTS) to
determine trajectory lengths dynamically.

The Hamiltonian uses an identity mass matrix. Instead of adapting a mass
matrix, BAT tunes the MCMC space transformation (see the `transform_tuning`
option of [`TransformedMCMC`](@ref)), which is mathematically equivalent.
Trajectory tuning is limited to the leapfrog step size (see
[`BAT.StepSizeAdaptor`](@ref)).

HMC uses gradients of the target measure's density, so your
[`BATContext`](@ref) needs to include an `ADSelector` to specify which
automatic differentiation backend should be used.

* Note: The fields of `HamiltonianMC` are still subject to change, and not
yet part of stable public BAT API!*

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct HamiltonianMC{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    SS<:Real,
    SJ<:Real,
    DE<:Real
} <: MCMCProposal
    target_acceptance::TA = 0.8
    target_acceptance_int::TAI = (0.9 * target_acceptance, one(Float64))

    "Leapfrog step size, `NaN` selects an automatic initial step size."
    step_size::SS = NaN

    "Relative random variation of the step size per transition."
    step_jitter::SJ = 0.0

    "Maximum NUTS trajectory tree depth."
    max_depth::Int = 10

    "Energy error above which a trajectory is considered divergent."
    max_delta_energy::DE = 1000.0
end

export HamiltonianMC


# Whole-run trajectory diagnostics of one chain, mutated in place so the
# counts survive the functional (immutable) proposal-state updates during
# tuning (all copies share this object by reference):
mutable struct HMCChainDiagnostics
    n_transitions::Int
    n_divergent::Int
    n_maxdepth::Int
    n_leapfrog::Int
    sum_p_accept::Float64
end

HMCChainDiagnostics() = HMCChainDiagnostics(0, 0, 0, 0, 0.0)

struct HMCProposalState{
    TA<:Real,
    TAI<:Tuple{Vararg{Real}},
    T<:AbstractFloat,
    FG<:Function
} <: MCMCProposalState
    target_acceptance::TA
    target_acceptance_int::TAI
    target_logdgrad::FG
    step_size::T
    step_jitter::T
    max_depth::Int
    max_delta_energy::T
    diagnostics::HMCChainDiagnostics
end


bat_default(::Type{TransformedMCMC}, ::Val{:pretransform}, proposal::HamiltonianMC) = NormalBased()

bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, proposal::HamiltonianMC) = StepSizeAdaptor()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, proposal::HamiltonianMC) = TriangularAffineTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, proposal::HamiltonianMC) = NoMCMCTempering()

bat_default(::Type{TransformedMCMC}, ::Val{:nsteps}, ::HamiltonianMC, ::TransformIntent, ::MCMCTransformTuning, nchains::Integer, nwalkers::Integer) = 10^4

bat_default(::Type{TransformedMCMC}, ::Val{:init}, ::HamiltonianMC, ::TransformIntent, ::MCMCTransformTuning, nchains::Integer, nwalkers::Integer, nsteps::Integer) =
    MCMCChainPoolInit(nsteps_init = 25)

bat_default(::Type{TransformedMCMC}, ::Val{:burnin}, ::HamiltonianMC, ::TransformIntent, ::MCMCTransformTuning, nchains::Integer, nwalkers::Integer, nsteps::Integer) =
    MCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 250), max_ncycles = 4)


function _hmc_target_logdgrad_func(target::BATMeasure, f_transform::Function, context::BATContext, proposal_alg)
    adsel = get_valid_adselector(context, proposal_alg)
    f = checked_logdensityof(MeasureBase.pullback(f_transform, target))
    return valgrad_func(f, adsel)
end

function _create_proposal_state(
    proposal::HamiltonianMC,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG
) where {P<:Real, PV<:AbstractVector{P}}
    @argcheck 0 < proposal.target_acceptance < 1
    let (lo, hi) = proposal.target_acceptance_int
        @argcheck 0 <= lo < hi <= 1
    end
    @argcheck isnan(proposal.step_size) || proposal.step_size > 0
    @argcheck 0 <= proposal.step_jitter < 1
    @argcheck proposal.max_depth >= 1
    @argcheck proposal.max_delta_energy > 0

    fg = _hmc_target_logdgrad_func(target, f_transform, context, proposal)

    T = float(P)
    step_size = if isnan(proposal.step_size)
        z_init = inverse(f_transform)(v_init[1])
        T(hmc_find_good_stepsize(rng, fg, z_init))
    else
        T(proposal.step_size)
    end

    return HMCProposalState(
        proposal.target_acceptance,
        proposal.target_acceptance_int,
        fg,
        step_size,
        T(proposal.step_jitter),
        Int(proposal.max_depth),
        T(proposal.max_delta_energy),
        HMCChainDiagnostics()
    )
end

function mcmc_propose!!(chain_state::MCMCChainState, proposal::HMCProposalState)
    (; f_transform, current, proposed, context) = chain_state
    n_walkers = nwalkers(chain_state)
    rng = get_rng(context)
    fg = proposal.target_logdgrad
    (; step_size, step_jitter, max_depth, max_delta_energy) = proposal
    T = typeof(step_size)

    p_accept = Vector{T}(undef, n_walkers)
    z_grads = Vector{Vector{T}}(undef, n_walkers)
    divergent = Vector{Bool}(undef, n_walkers)
    tree_depth = Vector{Int}(undef, n_walkers)
    n_leapfrog = Vector{Int}(undef, n_walkers)

    for i in 1:n_walkers
        q = convert(Vector{T}, current.z.v[i])
        z_current = _hmc_phasepoint(fg, q, randn(rng, T, length(q)))
        jittered_step_size = step_size * (1 + step_jitter * (2 * rand(rng, T) - 1))

        transition = hmc_nuts_transition(rng, fg, z_current, jittered_step_size, max_depth, max_delta_energy)

        proposed.z.v[i] = transition.z.q
        proposed.z.logd[i] = transition.z.logd
        p_accept[i] = transition.p_accept
        z_grads[i] = transition.z.grad
        divergent[i] = transition.divergent
        tree_depth[i] = transition.depth
        n_leapfrog[i] = transition.n_leapfrog
    end

    diag = proposal.diagnostics
    diag.n_transitions += n_walkers
    diag.n_divergent += count(divergent)
    diag.n_maxdepth += count(>=(max_depth), tree_depth)
    diag.n_leapfrog += sum(n_leapfrog)
    diag.sum_p_accept += sum(p_accept)

    chain_state.accepted .= proposed.z.v .!= current.z.v

    # proposed.z.logd already contains the pulled-back log-density, so the
    # x-space log-density follows from the transform's volume element alone:
    x_ladj_proposed = with_logabsdet_jacobian.(f_transform, proposed.z.v)
    proposed.x.v .= first.(x_ladj_proposed)
    ladj = getsecond.(x_ladj_proposed)
    proposed.x.logd .= proposed.z.logd .- ladj

    return chain_state, proposal, MCMCStepInfo(p_accept, z_grads, divergent, tree_depth, n_leapfrog)
end

mcmc_step_provides_grads(::HMCProposalState) = true

function set_proposal_transform!!(proposal::HMCProposalState, chain_state::MCMCChainState)
    fg = _hmc_target_logdgrad_func(chain_state.target, chain_state.f_transform, chain_state.context, proposal)
    return @set proposal.target_logdgrad = fg
end

function _proposal_diagnostics(p::HMCProposalState)
    d = p.diagnostics
    return (
        n_transitions = d.n_transitions,
        n_divergent = d.n_divergent,
        n_maxdepth = d.n_maxdepth,
        n_leapfrog = d.n_leapfrog,
        mean_p_accept = d.n_transitions > 0 ? d.sum_p_accept / d.n_transitions : NaN,
    )
end
