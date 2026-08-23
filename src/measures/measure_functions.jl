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


# ToDo: Replace try/catch-on-MethodError in the fallbacks below with a
# trait-based mechanism?
function _cov_with_fallback(d::UnivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    C = fill(T(NaN), n, n)
    try
        C[:] = Diagonal(fill(var(d),n))
    catch err
        if err isa MethodError
            C[:] = Diagonal(fill(var(VectorOfSimilarVectors(rand(rng, d, 10^5))),n))
        else
            throw(err)
        end
    end
    return C
end

function _cov_with_fallback(d::TDist, n::Integer)
    Σ = PDMat(Matrix(I(n) * one(Float64)))
end


function _cov_with_fallback(d::MultivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    C = fill(T(NaN), n, n)
    try
        C[:] = cov(d)
    catch err
        if err isa MethodError
            C[:] = cov(VectorOfSimilarVectors(rand(rng, d, 10^5)))
        else
            throw(err)
        end
    end
    return C
end

_approx_cov(target::Distribution, n) = _cov_with_fallback(target, n)
_approx_cov(target::BATDistMeasure, n) = _cov_with_fallback(Distribution(target), n)
_approx_cov(target::AbstractPosteriorMeasure, n) = _approx_cov(getprior(target), n)
_approx_cov(target::BATWeightedMeasure, n) = _approx_cov(basemeasure(target), n)
_approx_cov(target::BATMeasure, n) = cov(rand(_bat_determ_rng(), target^10^5))



function _mean_with_fallback(d::UnivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    m = fill(T(NaN), n)
    try
        m[:] = fill(mean(d),n)
    catch err
        if err isa MethodError
            m[:] = fill(mean(VectorOfSimilarVectors(rand(rng, d, 10^5))), n)
        else
            throw(err)
        end
    end
    return m
end

function _mean_with_fallback(d::TDist, n::Integer) # include arg for desired type of output?
    return ones(Float64, n) # technially only for degrees of freedom > 1
end


function _mean_with_fallback(d::MultivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    m = fill(T(NaN), n)
    try
        m[:] = mean(d)
    catch err
        if err isa MethodError
            m[:] = mean(VectorOfSimilarVectors(rand(rng, d, 10^5)))
        else
            throw(err)
        end
    end
    return m
end

_approx_mean(target::Distribution, n) = _mean_with_fallback(target, n)
_approx_mean(target::BATDistMeasure, n) = _mean_with_fallback(Distribution(target), n)
_approx_mean(target::AbstractPosteriorMeasure, n) = _approx_mean(getprior(target), n)
_approx_mean(target::BATWeightedMeasure, n) = _approx_mean(basemeasure(target), n)
_approx_mean(target::BATMeasure, n) = mean(rand(_bat_determ_rng(), target^10^5))
