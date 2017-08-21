# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions


"""
    AbstractProposalDist

The following functions must be implemented for subtypes:

* `BAT.proposal_logpdf!`
* `BAT.proposal_rand!`
* `Base.length`, returning the number of parameters (i.e. dimensionality).
* `Base.issymmetric`, indicating whether p(a -> b) == p(b -> a) holds true.
"""
abstract type AbstractProposalDist end
export AbstractProposalDist


"""
    proposal_logpdf!(
        p::AbstractArray,
        pdist::AbstractProposalDist,
        params_new::AbstractVecOrMat,
        params_old:::AbstractVecOrMat
    )

log(PDF) value of `pdist` for transitioning from old to new parameter values
for multiple parameter sets.
    
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

The caller must not assume that `proposal_logpdf!` is thread-safe.
"""
function proposal_logpdf! end
export proposal_logpdf!


# """
#     proposal_logpdf(
#         pdist::AbstractProposalDist,
#         params_new::AbstractVecOrMat,
#         params_old:::AbstractVecOrMat
#     )
# """
# function proposal_logpdf end
# export proposal_logpdf



"""
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
* `params_old`: Old parameter values (vector or column vectors in a matrix)

Output is stored in

* `params_new`: New parameter values (vector or column vectors in a matrix)

The caller must guarantee:

* `size(params_old, 1) == size(params_new, 1)`
* `size(params_old, 2) == size(params_new, 2)` or `size(params_old, 2) == 1`
* `params_new !== params_old` (no aliasing)

The caller must not assume that `proposal_rand!` is thread-safe.
"""
function proposal_rand! end
export proposal_rand!



struct GenericProposalDist{D<:Distribution,SamplerF,S<:Sampleable} <: AbstractProposalDist
    d::D
    sampler_f::SamplerF
    s::S

    function GenericProposalDist{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution,SamplerF}
        s = sampler_f(d)
        new{D,SamplerF, typeof(s)}(d, sampler_f, s)
    end

end

export GenericProposalDist

GenericProposalDist{D<:Distribution,SamplerF}(d::D, sampler_f::SamplerF) =
    GenericProposalDist{D,SamplerF}(d, sampler_f)

GenericProposalDist(d::Distribution) = GenericProposalDist(d, bat_sampler)

Base.similar(q::GenericProposalDist, d::Distribution) =
    GenericProposalDist(d, q.sampler_f)


function proposal_logpdf!(
    p::AbstractArray,
    pdist::GenericProposalDist,
    params_new::AbstractMatrix,
    params_old::AbstractVecOrMat
)
    params_diff = params_new .- params_old # TODO: Avoid memory allocation
    Distributions.logpdf!(p, pdist.d, params_diff)
end


function proposal_logpdf!(
    p::AbstractArray,
    pdist::GenericProposalDist,
    params_new::AbstractVector,
    params_old::AbstractVector
)
    params_diff = params_new .- params_old # TODO: Avoid memory allocation
    p[1] = Distributions.logpdf(pdist.d, params_diff)
    p
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


Base.length(pdist::GenericProposalDist) = length(pdist.d)

Base.issymmetric(pdist::GenericProposalDist) = issymmetric_around_origin(pdist.d)
