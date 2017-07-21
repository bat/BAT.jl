# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions

"""
    AbstractProposalFunction

Subtypes (e.g. `ProposalFunc <: AbstractProposalFunction`) must implement

```
rand!(
    rng::AbstractRNG,
    f::ProposalFunc,
    new_params::Matrix,
    old_params:::Matrix,
    bounds::AbstractParamBounds
)::Void
```

Proposal functions must make a single attemt to generate valid new vector of
parameters with a random offset to `old_params` (e.g. based on an internal
probability distribution). If the new parameter vector is in `bounds`,
it is to be written into `new_params`. Otherwise, the proposal function may
either:

* Fill `new_params` with `NaN` values (for floating-point parameters), resp.
  `0xFFFF...` values (for integer parameters)
* Remap the parameters into bounds (e.g. by assuming a cyclic parameter
  space)

The handling of out-of-bounds parameters must be consistent and should
ideally be configurable.

The caller must guarantee:

* ```indices(`new_params`) == indices(`old_params`)```
* `new_params !== old_params`, no aliasing
* For a specific instance, `rand!` is always called on the same thread.

"""
abstract type AbstractProposalFunction end


#=

Alternative: Functions `invalidate_params!(params::Matrix, i::Integer)` and




export ProposalFunction

struct ProposalFunction{D<:Distribution,SamplerF} <: AbstractProposalFunction
    d::D
    sampler_f::SamplerF

    function ProposalFunction{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution,SamplerF}
        issymmetric_at_origin(d) || throw(ArgumentError("Distribution $d must be symmetric at origin"))
        new{D,SamplerF}(d, sampler_f)
    end

end

ProposalFunction{D<:Distribution,SamplerF}(d::D, sampler_f::SamplerF) = ProposalFunction{D,SamplerF}(d, sampler_f)

ProposalFunction(d::Distribution) = ProposalFunction(d, bat_sampler)

Distributions.sampler(q::ProposalFunction) = q.sampler_f(q.d)

=#
