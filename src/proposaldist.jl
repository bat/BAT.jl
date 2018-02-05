# This file is a part of BAT.jl, licensed under the MIT License (MIT).


doc"""
    AbstractProposalDist

The following functions must be implemented for subtypes:

* `BAT.distribution_logpdf`
* `BAT.proposal_rand!`
* `BAT.nparams`, returning the number of parameters (i.e. dimensionality).
* `Base.issymmetric`, indicating whether p(a -> b) == p(b -> a) holds true.

In some cases, it may be desirable to override the default implementation
of `BAT.distribution_logpdf!`.
"""
abstract type AbstractProposalDist end
export AbstractProposalDist


doc"""
    distribution_logpdf(
        pdist::AbstractProposalDist,
        params_new::AbstractVector,
        params_old:::AbstractVector
    )

Analog to `distribution_logpdf!`, but for a single parameter vector.
"""
function distribution_logpdf end
export distribution_logpdf

# TODO: Implement distribution_logpdf for included proposal distributions


doc"""
    distribution_logpdf!(
        p::AbstractArray,
        pdist::AbstractProposalDist,
        params_new::AbstractVecOrMat,
        params_old:::AbstractVecOrMat
    )

Returns log(PDF) value of `pdist` for transitioning from old to new parameter
values for multiple parameter sets.

end

Input:

* `params_new`: New parameter values (column vectors)
* `params_old`: Old parameter values (column vectors)

Output is stored in

* `p`: Array of PDF values, length must match, shape is ignored

Array size requirements:

* `size(params_old, 1) == size(params_new, 1) == length(pdist)`
* `size(params_old, 2) == size(params_new, 2)` or `size(params_old, 2) == 1`
* `size(params_new, 2) == length(p)`

Implementations of `distribution_logpdf!` must be thread-safe.
"""
function distribution_logpdf! end
export distribution_logpdf!

# TODO: Default implementation of distribution_logpdf!


doc"""
    function proposal_rand!(
        rng::AbstractRNG,
        pdist::GenericProposalDist,
        params_new::AbstractVecOrMat,
        params_old::AbstractVecOrMat
    )

Generate one or multiple proposed parameter vectors, based on one or multiple
previous parameter vectors.

Input:

* `rng`: Random number generator to use
* `pdist`: Proposal distribution to use
* `params_old`: Old parameter values (vector or column vectors, if a matrix)

Output is stored in

* `params_new`: New parameter values (vector or column vectors, if a matrix)

The caller must guarantee:

* `size(params_old, 1) == size(params_new, 1)`
* `size(params_old, 2) == size(params_new, 2)` or `size(params_old, 2) == 1`
* `params_new !== params_old` (no aliasing)

Implementations of `proposal_rand!` must be thread-safe.
"""
function proposal_rand! end
export proposal_rand!



struct GenericProposalDist{D<:Distribution{Multivariate},SamplerF,S<:Sampleable} <: AbstractProposalDist
    d::D
    sampler_f::SamplerF
    s::S

    function GenericProposalDist{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF}
        s = sampler_f(d)
        new{D,SamplerF, typeof(s)}(d, sampler_f, s)
    end

end

export GenericProposalDist

GenericProposalDist(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF} =
    GenericProposalDist{D,SamplerF}(d, sampler_f)

GenericProposalDist(d::Distribution{Multivariate}) = GenericProposalDist(d, bat_sampler)

GenericProposalDist(D::Type{<:Distribution{Multivariate}}, n_params::Integer, args...) =
    GenericProposalDist(D, Float64, n_params, args...)


Base.similar(q::GenericProposalDist, d::Distribution{Multivariate}) =
    GenericProposalDist(d, q.sampler_f)

function Base.convert(::Type{AbstractProposalDist}, q::GenericProposalDist, T::Type{<:AbstractFloat}, n_params::Integer)
    n_params != nparams(q) && throw(ArgumentError("q has wrong number of parameters"))
    q
end


get_cov(q::GenericProposalDist) = get_cov(q.d)
set_cov!(q::GenericProposalDist, Σ::AbstractMatrix{<:Real}) = similar(q, set_cov!(q.d, Σ))


function distribution_logpdf!(
    p::AbstractArray,
    pdist::GenericProposalDist,
    params_new::AbstractMatrix,
    params_old::AbstractVecOrMat
)
    params_diff = params_new .- params_old # TODO: Avoid memory allocation
    Distributions.logpdf!(p, pdist.d, params_diff)
end


function distribution_logpdf!(
    p::AbstractArray,
    pdist::GenericProposalDist,
    params_new::AbstractVector,
    params_old::AbstractVector
)
    p[1] = distribution_logpdf(pdist, params_new, params_old)
    p
end


function distribution_logpdf(
    pdist::GenericProposalDist,
    params_new::AbstractVector,
    params_old::AbstractVector
)
    params_diff = params_new .- params_old # TODO: Avoid memory allocation
    Distributions.logpdf(pdist.d, params_diff)
end


function proposal_rand!(
    rng::AbstractRNG,
    pdist::GenericProposalDist,
    params_new::AbstractVecOrMat,
    params_old::AbstractVecOrMat
)
    rand!(rng, pdist.s, params_new)
    params_new .+= params_old
end


nparams(pdist::GenericProposalDist) = length(pdist.d)

Base.issymmetric(pdist::GenericProposalDist) = issymmetric_around_origin(pdist.d)




abstract type ProposalDistSpec end

export ProposalDistSpec


struct MvTDistProposalSpec <: ProposalDistSpec
    df::Float64
end

export MvTDistProposalSpec

MvTDistProposalSpec() = MvTDistProposalSpec(1.0)


(ps::MvTDistProposalSpec)(T::Type{<:AbstractFloat}, n_params::Integer) =
    GenericProposalDist(MvTDist, T, n_params, convert(T, ps.df))

function GenericProposalDist(::Type{MvTDist}, T::Type{<:AbstractFloat}, n_params::Integer, df = one(T))
    Σ = PDMat(full(ScalMat(n_params, one(T))))
    zeromean = true
    μ = fill(zero(T), n_params)
    M = typeof(Σ)
    d = Distributions.GenericMvTDist{T,M}(convert(T, df), n_params, zeromean, μ, Σ)
    GenericProposalDist(d)
end
