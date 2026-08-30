# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct StretchMove <: MCMCProposal

Experimental affine-invariant ensemble proposal using the Goodman--Weare
stretch move. `nwalkers` must be set explicitly on `TransformedMCMC`.

Constructors:

* ```$(FUNCTIONNAME)(; scale = 2)```
"""
struct StretchMove{S<:Real} <: MCMCProposal
    scale::S

    function StretchMove(scale::S) where {S<:Real}
        isfinite(scale) && scale > one(scale) || throw(ArgumentError(
            "StretchMove scale must be finite and greater than 1",
        ))
        new{S}(scale)
    end
end
StretchMove(; scale::Real = 2) = StretchMove(scale)
export StretchMove


struct StretchMoveProposalState{S<:Real} <: MCMCProposalState
    scale::S
end

_mcmc_n_rng_purposes(::StretchMoveProposalState) = _MCMC_N_RNG_PURPOSES


bat_default(::Type{TransformedMCMC}, ::Val{:proposal_tuning}, ::StretchMove) =
    NoMCMCProposalTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:adaptive_transform}, ::StretchMove) =
    NoAdaptiveTransform()

bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::StretchMove, ::NoAdaptiveTransform) =
    NoMCMCTransformTuning()

bat_default(::Type{TransformedMCMC}, ::Val{:tempering}, ::StretchMove) =
    NoMCMCTempering()

function bat_default(
    ::Type{TransformedMCMC},
    ::Val{:nwalkers},
    ::StretchMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
)
    throw(ArgumentError("StretchMove requires an explicit nwalkers setting on TransformedMCMC"))
end

bat_default(
    ::Type{TransformedMCMC},
    ::Val{:init},
    ::StretchMove,
    ::TransformIntent,
    ::MCMCTransformTuning,
    ::Integer,
    ::Integer,
    ::Integer,
) = MCMCRetryInit()


_validate_mcmc_proposal_configuration(
    ::StretchMove,
    ::NoMCMCProposalTuning,
) = nothing

function _validate_mcmc_proposal_configuration(
    ::StretchMove,
    tuning::MCMCProposalTuning,
)
    throw(ArgumentError(
        "StretchMove requires NoMCMCProposalTuning, got $(nameof(typeof(tuning)))",
    ))
end

_validate_stretch_move_transform_tuning(
    ::StretchMove,
    ::NoMCMCTransformTuning,
) = nothing

function _validate_stretch_move_transform_tuning(
    ::StretchMove,
    tuning::MCMCTransformTuning,
)
    throw(ArgumentError(
        "StretchMove requires NoMCMCTransformTuning, got $(nameof(typeof(tuning)))",
    ))
end


_validate_mcmc_adaptive_transform_configuration(
    ::StretchMove,
    ::NoAdaptiveTransform,
) = nothing

function _validate_mcmc_adaptive_transform_configuration(
    ::StretchMove,
    adaptive_transform::AbstractAdaptiveTransform,
)
    throw(ArgumentError(
        "StretchMove requires NoAdaptiveTransform, got $(nameof(typeof(adaptive_transform)))",
    ))
end


function _validate_mcmc_weighting_configuration(
    ::StretchMove,
    ::RepetitionWeighting,
)
    return nothing
end

function _validate_mcmc_weighting_configuration(
    ::StretchMove,
    weighting::AbstractMCMCWeightingScheme,
)
    throw(ArgumentError(
        "StretchMove supports RepetitionWeighting only, got $(nameof(typeof(weighting)))",
    ))
end


function _create_proposal_state(
    proposal::StretchMove,
    target::BATMeasure,
    context::BATContext,
    v_init::AbstractVector{PV},
    f_transform::Function,
    rng::AbstractRNG,
) where {P<:Real, PV<:AbstractVector{P}}
    n_walkers = length(v_init)
    n_dims = totalndof(varshape(target))
    n_walkers >= 2 * n_dims || throw(ArgumentError(
        "StretchMove requires at least 2 * d walkers; got $n_walkers walkers for dimension $n_dims",
    ))

    z_init = inverse(f_transform).(v_init)
    scale_type = float(eltype(first(z_init)))
    scale = convert(scale_type, proposal.scale)
    isfinite(scale) && scale > one(scale) || throw(ArgumentError(
        "StretchMove scale must remain finite and greater than 1 after conversion to $(scale_type)",
    ))
    all(z -> all(isfinite, z), z_init) || throw(ArgumentError(
        "StretchMove requires finite transformed coordinates during initialization",
    ))

    centered_z = reduce(hcat, map(z -> z .- first(z_init), z_init))
    # `rank` compares singular values with `rtol * σ₁`; scaling rtol by the
    # matrix dimensions and scalar precision therefore preserves the affine
    # rank decision under a common coordinate rescaling.
    rank_rtol = max(size(centered_z)...)*eps(float(real(eltype(centered_z))))
    observed_rank = rank(centered_z; rtol = rank_rtol)
    observed_rank == n_dims || throw(ArgumentError(
        "StretchMove initialization has $n_walkers walkers in dimension $n_dims with affine rank $observed_rank; expected affine rank $n_dims",
    ))

    return StretchMoveProposalState(scale)
end


_stretch_scale(scale::Real, u::Real) =
    ((scale - one(scale)) * u + one(scale))^2 / scale

_stretch_candidate(current, companion, scale::Real) =
    companion + scale * (current - companion)

_stretch_log_acceptance(
    n_dims::Integer, scale::Real, proposed_logd::Real, current_logd::Real,
) = (n_dims - 1) * log(scale) + proposed_logd - current_logd


function _stretch_move_groups(
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_order::AbstractVector{<:Integer},
)
    split_rng = AbstractRNG(
        step_rngpart,
        _mcmc_rng_stream_idx(_MCMC_ENSEMBLE_SPLIT_PURPOSE, proposal_idx),
    )
    permutation = randperm(split_rng, length(walker_order))
    split_at = fld(length(walker_order), 2)
    left = walker_order[permutation[begin:split_at]]
    right = walker_order[permutation[(split_at + 1):end]]
    return rand(split_rng, Bool) ? (left, right) : (right, left)
end


function _propose_stretch_subset!!(
    chain_state::MCMCChainState,
    proposal::StretchMoveProposalState,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_idxs::AbstractVector{<:Integer},
    companion_idxs::AbstractVector{<:Integer},
    p_accept::AbstractVector{<:Real},
)
    (;target, f_transform, current, proposed) = chain_state
    companion_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_COMPANION_SELECTION_PURPOSE, proposal_idx,
    )
    stretch_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_STRETCH_DRAW_PURPOSE, proposal_idx,
    )
    acceptance_rngpart = _mcmc_walker_rngpart(
        step_rngpart, _MCMC_ACCEPTANCE_PURPOSE, proposal_idx,
    )
    constant_ladj = _transform_ladj(f_transform)

    for i in walker_idxs
        walkerid = current.x.info[i].walkerid
        rng = get_rng(chain_state.walker_genctxs[i])

        set_rng!(rng, companion_rngpart, walkerid)
        companion_idx = rand(rng, companion_idxs)

        set_rng!(rng, stretch_rngpart, walkerid)
        T = float(eltype(current.z.v[i]))
        stretch = _stretch_scale(proposal.scale, rand(rng, T))
        z_proposed = _stretch_candidate(
            current.z.v[i], current.z.v[companion_idx], stretch,
        )

        x_proposed, ladj = if isnothing(constant_ladj)
            with_logabsdet_jacobian(f_transform, z_proposed)
        else
            f_transform(z_proposed), constant_ladj
        end
        logd_x_proposed = BAT.checked_logdensityof(target, x_proposed)
        logd_z_proposed = logd_x_proposed + ladj

        proposed.x.v[i] .= x_proposed
        proposed.z.v[i] .= z_proposed
        proposed.x.logd[i] = logd_x_proposed
        proposed.z.logd[i] = logd_z_proposed

        log_acceptance = _stretch_log_acceptance(
            length(z_proposed), stretch, logd_z_proposed, current.z.logd[i],
        )
        p_accept[i] = _mcmc_acceptance_probability(convert(T, log_acceptance))
        set_rng!(rng, acceptance_rngpart, walkerid)
        chain_state.accepted[i] = rand(rng, T) < p_accept[i]
    end

    return chain_state
end


function _mcmc_step_transition!!(
    mcmc_state::MCMCState,
    active_proposal::StretchMoveProposalState,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walker_order::AbstractVector{<:Integer},
)
    chain_state = mcmc_state.chain_state
    first_group, second_group = _stretch_move_groups(
        step_rngpart, proposal_idx, walker_order,
    )
    T = float(eltype(first(chain_state.current.z.v)))
    p_accept = Vector{T}(undef, length(walker_order))
    step_info = MCMCStepInfo(
        p_accept, nothing, nothing, nothing, nothing, walker_order,
    )

    _propose_stretch_subset!!(
        chain_state, active_proposal, step_rngpart, proposal_idx,
        first_group, second_group, p_accept,
    )
    _apply_mcmc_subset!!(chain_state, step_info, first_group)

    _propose_stretch_subset!!(
        chain_state, active_proposal, step_rngpart, proposal_idx,
        second_group, first_group, p_accept,
    )
    _apply_mcmc_subset!!(chain_state, step_info, second_group)

    chain_state.proposal = update_active_proposal!!(
        chain_state.proposal, active_proposal,
    )
    mcmc_state_new = mcmc_tune_post_step!!(
        mcmc_state, active_proposal, step_info,
    )
    chain_state = mcmc_state_new.chain_state
    active_prop_idx = get_active_proposal_idx(chain_state.proposal)
    chain_state.nattempts[active_prop_idx] += length(chain_state.accepted)
    chain_state.nsamples[active_prop_idx] += sum(chain_state.accepted)

    return @set mcmc_state_new.chain_state = chain_state
end
