# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type TransformedAbstractProposalDist

*BAT-internal, not part of stable public API.*

The following functions must be implemented for subtypes:

* `BAT.proposaldist_logpdf`
* `BAT.proposal_rand!`
* `ValueShapes.totalndof`, returning the number of DOF (i.e. dimensionality).
* `LinearAlgebra.issymmetric`, indicating whether p(a -> b) == p(b -> a) holds true.
"""
abstract type TransformedAbstractProposalDist end

# TODO AC: reactivate
# """
#     proposaldist_logpdf(
#         p::AbstractArray,
#         pdist::TransformedAbstractProposalDist,
#         v_proposed::AbstractVector,
#         v_current:::AbstractVector
#     )

# *BAT-internal, not part of stable public API.*

# Returns log(PDF) value of `pdist` for transitioning from current to proposed
# variate/parameters.
# """#function proposaldist_logpdf end

# TODO: Implement proposaldist_logpdf for included proposal distributions


# TODO AC: reactivate
# """
#     function proposal_rand!(
#         rng::AbstractRNG,
#         pdist::TransformedGenericProposalDist,
#         v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
#         v_current::Union{AbstractVector,VectorOfSimilarVectors}
#     )

# *BAT-internal, not part of stable public API.*

# Generate one or multiple proposed variate/parameter vectors, based on one or
# multiple previous vectors.

# Input:

# * `rng`: Random number generator to use
# * `pdist`: Proposal distribution to use
# * `v_current`: Old values (vector or column vectors, if a matrix)

# Output is stored in

# * `v_proposed`: New values (vector or column vectors, if a matrix)

# The caller must guarantee:

# * `size(v_current, 1) == size(v_proposed, 1)`
# * `size(v_current, 2) == size(v_proposed, 2)` or `size(v_current, 2) == 1`
# * `v_proposed !== v_current` (no aliasing)

# Implementations of `proposal_rand!` must be thread-safe.
# """
# function proposal_rand! end



struct TransformedGenericProposalDist{D<:Distribution{Multivariate},SamplerF,S<:Sampleable} <: TransformedAbstractProposalDist
    d::D
    sampler_f::SamplerF
    s::S

    function TransformedGenericProposalDist{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF}
        s = sampler_f(d)
        new{D,SamplerF, typeof(s)}(d, sampler_f, s)
    end

end


TransformedGenericProposalDist(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF} =
    TransformedGenericProposalDist{D,SamplerF}(d, sampler_f)

TransformedGenericProposalDist(d::Distribution{Multivariate}) = TransformedGenericProposalDist(d, bat_sampler)

TransformedGenericProposalDist(D::Type{<:Distribution{Multivariate}}, varndof::Integer, args...) =
    TransformedGenericProposalDist(D, Float64, varndof, args...)


Base.similar(q::TransformedGenericProposalDist, d::Distribution{Multivariate}) =
    TransformedGenericProposalDist(d, q.sampler_f)

function Base.convert(::Type{TransformedAbstractProposalDist}, q::TransformedGenericProposalDist, T::Type{<:AbstractFloat}, varndof::Integer)
    varndof != totalndof(q) && throw(ArgumentError("q has wrong number of DOF"))
    q
end


get_cov(q::TransformedGenericProposalDist) = get_cov(q.d)
set_cov(q::TransformedGenericProposalDist, Σ::PosDefMatLike) = similar(q, set_cov(q.d, Σ))


function proposaldist_logpdf(
    pdist::TransformedGenericProposalDist,
    v_proposed::AbstractVector,
    v_current::AbstractVector
)
    params_diff = v_proposed .- v_current # TODO: Avoid memory allocation
    logpdf(pdist.d, params_diff)
end


function proposal_rand!(
    rng::AbstractRNG,
    pdist::TransformedGenericProposalDist,
    v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
    v_current::Union{AbstractVector,VectorOfSimilarVectors}
)
    rand!(rng, pdist.s, flatview(v_proposed))
    params_new_flat = flatview(v_proposed)
    params_new_flat .+= flatview(v_current)
    v_proposed
end


ValueShapes.totalndof(pdist::TransformedGenericProposalDist) = length(pdist.d)

LinearAlgebra.issymmetric(pdist::TransformedGenericProposalDist) = issymmetric_around_origin(pdist.d)



struct TransformedGenericUvProposalDist{D<:Distribution{Univariate},T<:Real,SamplerF,S<:Sampleable} <: TransformedAbstractProposalDist
    d::D
    scale::Vector{T}
    sampler_f::SamplerF
    s::S
end


TransformedGenericUvProposalDist(d::Distribution{Univariate}, scale::Vector{<:AbstractFloat}, samplerF) =
    TransformedGenericUvProposalDist(d, scale, samplerF, samplerF(d))

TransformedGenericUvProposalDist(d::Distribution{Univariate}, scale::Vector{<:AbstractFloat}) =
    TransformedGenericUvProposalDist(d, scale, bat_sampler)


ValueShapes.totalndof(pdist::TransformedGenericUvProposalDist) = size(pdist.scale, 1)

LinearAlgebra.issymmetric(pdist::TransformedGenericUvProposalDist) = issymmetric_around_origin(pdist.d)

function BAT.proposaldist_logpdf(
    pdist::TransformedGenericUvProposalDist,
    v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
    v_current::Union{AbstractVector,VectorOfSimilarVectors}
)
    params_diff = (flatview(v_proposed) .- flatview(v_current)) ./ pdist.scale  # TODO: Avoid memory allocation
    sum_first_dim(logpdf.(pdist.d, params_diff))  # TODO: Avoid memory allocation
end

function BAT.proposal_rand!(
    rng::AbstractRNG,
    pdist::TransformedGenericUvProposalDist,
    v_proposed::AbstractVector,
    v_current::AbstractVector
)
    v_proposed .= v_current
    dim = rand(rng, eachindex(pdist.scale))
    v_proposed[dim] += pdist.scale[dim] * rand(rng, pdist.s)
    v_proposed
end



abstract type TransformedProposalDistSpec end


struct TransformedMvTDistProposal <: TransformedProposalDistSpec
    df::Float64
end

TransformedMvTDistProposal() = TransformedMvTDistProposal(1.0)


(ps::TransformedMvTDistProposal)(T::Type{<:AbstractFloat}, varndof::Integer) =
    TransformedGenericProposalDist(MvTDist, T, varndof, convert(T, ps.df))

function TransformedGenericProposalDist(::Type{MvTDist}, T::Type{<:AbstractFloat}, varndof::Integer, df = one(T))
    Σ = PDMat(Matrix(ScalMat(varndof, one(T))))
    μ = Fill(zero(eltype(Σ)), varndof)
    M = typeof(Σ)
    d = Distributions.GenericMvTDist(convert(T, df), μ, Σ)
    TransformedGenericProposalDist(d)
end


struct TransformedUvTDistProposalSpec <: TransformedProposalDistSpec
    df::Float64
end

(ps::TransformedUvTDistProposalSpec)(T::Type{<:AbstractFloat}, varndof::Integer) =
    TransformedGenericUvProposalDist(TDist(convert(T, ps.df)), fill(one(T), varndof))
