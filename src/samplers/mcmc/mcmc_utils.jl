# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _cov_with_fallback(d::UnivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    C = fill(T(NaN), n, n)
    try
        C[:] = Diagonal(fill(var(d),n))
    catch err
        if err isa MethodError
            C[:] = Diagonal(fill(var(nestedview(rand(rng, d, 10^5))),n))
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
            C[:] = cov(nestedview(rand(rng, d, 10^5)))
        else
            throw(err)
        end
    end
    return C
end

_approx_cov(target::Distribution, n) = _cov_with_fallback(target, n)
_approx_cov(target::BATDistMeasure, n) = _cov_with_fallback(Distribution(target), n)
_approx_cov(target::AbstractPosteriorMeasure, n) = _approx_cov(getprior(target), n)
