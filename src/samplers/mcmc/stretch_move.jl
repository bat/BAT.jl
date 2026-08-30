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
