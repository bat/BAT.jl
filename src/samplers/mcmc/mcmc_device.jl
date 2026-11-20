# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# MCMC chains whose walkers live on a compute unit other than the host.


# Chains on such compute units support a subset of the MCMC components so far:
function _check_device_mcmc_support(samplingalg::TransformedMCMC)
    (; proposal, proposal_tuning, adaptive_transform, transform_tuning, tempering, sample_weighting) = samplingalg
    _device_mcmc_requires(
        proposal isa RandomWalk && proposal.proposaldist isa Union{Normal,TDist},
        proposal, "a RandomWalk proposal with normal or t-distributed innovations"
    )
    _device_mcmc_requires(proposal_tuning isa NoMCMCProposalTuning, proposal_tuning, "NoMCMCProposalTuning")
    _device_mcmc_requires(
        adaptive_transform isa Union{NoAdaptiveTransform,TriangularAffineTransform},
        adaptive_transform, "TriangularAffineTransform or NoAdaptiveTransform"
    )
    _device_mcmc_requires(
        transform_tuning isa Union{NoMCMCTransformTuning,RAMTuning}, transform_tuning, "RAMTuning or NoMCMCTransformTuning"
    )
    _device_mcmc_requires(tempering isa NoMCMCTempering, tempering, "NoMCMCTempering")
    _device_mcmc_requires(sample_weighting isa RepetitionWeighting, sample_weighting, "RepetitionWeighting")
    return nothing
end

function _device_mcmc_requires(supported::Bool, @nospecialize(component), requirement::AbstractString)
    supported || throw(ArgumentError(
        "MCMC sampling on compute units other than the host requires $requirement, got $(nameof(typeof(component)))"
    ))
    return nothing
end


# An array on the compute unit, with real numbers of type `T`:
_cunit_array(cunit::AbstractComputeUnit, ::Type{T}, A::AbstractArray{<:Real}) where {T<:Real} =
    adapt(cunit, convert(AbstractArray{T}, A))

_device_batch(cunit::AbstractComputeUnit, ::Type{T}, X::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real} =
    nestedview(_cunit_array(cunit, T, flatview(VectorOfSimilarVectors(X))))

_device_transform(::AbstractComputeUnit, ::Type{<:Real}, ::typeof(identity)) = identity

# A dense matrix with zeros above the diagonal holds the triangular factor,
# so that products with it can also use dense matrix multiplication:
_device_transform(cunit::AbstractComputeUnit, ::Type{T}, f::MulAdd{<:LowerTriangular}) where {T<:Real} =
    MulAdd(LowerTriangular(_cunit_array(cunit, T, Matrix(f.A))), _cunit_array(cunit, T, f.b))
