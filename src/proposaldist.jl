# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions

"""
    ProposalDist

The following functions must be implemented for subtypes:

* `proposal_pdf!`
* `proposal_rand!`
"""
abstract type ProposalDist end


"""
    proposal_pdf!(
        pdist::ProposalDist,
        p::AbstractVector
        new_params::AbstractMatrix,
        old_params:::AbstractMatrix,
        exec_context::ExecContext = ExecContext()
    )

PDF value of `pdist` for transitioning from old to new parameter values for
multiple parameter sets.
    
end

Input:

* `new_params`: New parameter values (column vectors)
* `old_params`: Old parameter values (column vectors)

Output is stored in

* `p`: Vector of PDF values

The caller must not assume that `proposal_pdf!` is thread-safe. Depending
on `exec_context` it may also use multiple threads/processes internally.
"""
function proposal_pdf! end
export proposal_pdf!



"""
    function proposal_rand!(
        rng::AbstractRNG,
        pdist::ProposalDist,
        new_params::AbstractMatrix,
        old_params::AbstractMatrix,
        bounds::AbstractParamBounds,
        exec_context::ExecContext = ExecContext()
    )

For each column of `old_params`, make a single attemt to generate valid new
vector of parameters based on `pdist`, given `old_params`. Ensures that each
new parameter vector is either withing `bounds`, or marked invalid by
settings at least one of its elements to `oob(eltype(new_params))`.

Input:

* `rng`: Random number generator to use
* `pdist`: Proposal distribution to use
* `old_params`: Old parameter values (column vectors)
* `bounds`: Parameter bounds (column vectors)

Output is stored in

* `new_params`: New parameter values (column vectors)

The caller must guarantee:

* ```indices(`new_params`) == indices(`old_params`)```
* `new_params !== old_params`, no aliasing
* For a specific instance, `rand!` is always called on the same thread.

The caller must not assume that `proposal_pdf_rand!` is thread-safe. Depending
on `exec_context` it may also use multiple threads/processes internally.
"""
function proposal_rand! end
export proposal_rand!


#=
"""
    function proposal_pdf_rand!(
        rng::AbstractRNG,
        pdist::ProposalDist,
        p_fwd::AbstractVector,
        p_bck::AbstractVector,
        new_params::AbstractMatrix,
        old_params::AbstractMatrix,
        bounds::AbstractParamBounds,
        exec_context::ExecContext = ExecContext()
    )

Combines `proposal_pdf!` and `proposal_rand!`.
"""

function proposal_pdf_rand!(
    rng::AbstractRNG,
    pdist::ProposalDist,
    p_fwd::AbstractVector,
    p_bck::AbstractVector,
    new_params::AbstractMatrix,
    old_params::AbstractMatrix,
    bounds::AbstractParamBounds,
    exec_context::ExecContext = ExecContext()
)
    proposal_pdf!(p_fwd, pdist, new_params, old_params)
    proposal_pdf!(p_bck, pdist, old_params, new_params)
    proposal_rand!(rng, pdist, new_params, old_params, bounds, exec_context)
end

export proposal_pdf_rand!
=#



struct GenericProposalDist{D<:Distribution,SamplerF,S<:Sampleable} <: ProposalDist
    d::D
    sampler_f::SamplerF
    s::S

    function GenericProposalDist{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution,SamplerF}
        issymmetric_at_origin(d) || throw(ArgumentError("Distribution $d must be symmetric at origin"))
        s = sampler_f(d)
        new{D,SamplerF, typeof(s)}(d, sampler_f, s)
    end

end

export GenericProposalDist

GenericProposalDist{D<:Distribution,SamplerF}(d::D, sampler_f::SamplerF) = GenericProposalDist{D,SamplerF}(d, sampler_f)

GenericProposalDist(d::Distribution) = GenericProposalDist(d, bat_sampler)

Base.similar(q::GenericProposalDist, d::Distribution) =
    GenericProposalDist(d, q.sampler_f)
