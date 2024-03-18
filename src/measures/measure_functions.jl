# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    distprod(;a = some_dist, b = some_other_dist, ...)
    distprod(();a = some_dist, b = some_other_dist, ...))
    distprod([dist1, dist2, dist2, ...])

Generate a product of distributions, returning either a distribution
that has NamedTuples as variates, or arrays as variates.
"""
function distprod end
export distprod

@inline distprod(ds::NamedTuple) = ValueShapes.NamedTupleDist(ds)
@inline distprod(;kwargs...) = ValueShapes.NamedTupleDist(;kwargs...)
@inline distprod(Ds::AbstractArray) = Distributions.product_distribution(Ds)


"""
    lbqintegral(integrand, measure)
    lbqintegral(likelihood, prior)

Returns an object that represents the Lebesgue integral over a function
in respect to s reference measure. It is also the non-normalized
posterior measure that results from integrating the likelihood of
a given observation in respect to a prior measure.
"""
function lbqintegral end
export lbqintegral

@inline lbqintegral(integrand, measure) = PosteriorMeasure(integrand, batmeasure(measure))


"""
    distbind(f_k, dist, ::typeof(merge))

Performs a generalized monadic bind, in the functional programming sense,
with a transition kernel `f_k`, a distribution `dist`, using `merge` to
control the type of "flattening".
"""
function distbind end
export distbind

function distbind(f_k, dist::Distribution, ::typeof(merge))
    @argcheck dist isa NamedTupleDist
    HierarchicalDistribution(f_k, dist)
end

function distbind(f_k, dist::Distribution, ::typeof(vcat))
    @argcheck dist isa Union{UnivariateDistribution, MultivariateDistribution}
    HierarchicalDistribution(f_k, dist)
end
