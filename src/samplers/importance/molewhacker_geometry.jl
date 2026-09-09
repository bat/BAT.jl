# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Pull back distribution Fisher information without constructing its parameter-space
# matrix. In particular, the covariance block of an m-dimensional normal would
# otherwise require O(m^4) storage.
struct MolewhackerGeometryError <: Exception end

function _mw_regular_positive(x)
    isfinite(x) && x > 0 || throw(MolewhackerGeometryError())
    return x
end

_mw_parameters(d::Normal) = collect(params(d))
_mw_parameters(d::Union{Poisson,Exponential}) = [mean(d)]
_mw_parameters(d::MvNormal) = vcat(mean(d), vec(Matrix(cov(d))))

_mw_factors(d::Distributions.Product) = d.v
_mw_factors(d::Distributions.ProductDistribution) = vec(d.dists)
_mw_factors(d::NamedTupleDist) = values(d)

function _mw_parameters(d::Union{Distributions.Product,Distributions.ProductDistribution,NamedTupleDist})
    return reduce(vcat, map(_mw_parameters, _mw_factors(d)))
end

_mw_nparams(::Normal) = 2
_mw_nparams(::Union{Poisson,Exponential}) = 1
_mw_nparams(d::MvNormal) = length(d) + length(d)^2
function _mw_nparams(d::Union{Distributions.Product,Distributions.ProductDistribution,NamedTupleDist})
    return sum(_mw_nparams, _mw_factors(d))
end

function _mw_parameters(d)
    throw(ArgumentError("MolewhackerSampling has no Fisher geometry for $(typeof(d)). Supported families are Normal, MvNormal, Poisson, Exponential, and their products."))
end

function _mw_pullback(d::Normal, J)
    σ = _mw_regular_positive(std(d))
    A = J ./ σ
    return A[1:1, :]' * A[1:1, :] + 2 .* (A[2:2, :]' * A[2:2, :])
end

function _mw_pullback(d::Union{Poisson,Exponential}, J)
    m = _mw_regular_positive(mean(d))
    scale = d isa Poisson ? sqrt(m) : m
    A = J ./ scale
    return A' * A
end

function _mw_pullback(d::MvNormal, J)
    m = length(d)
    Σ = Matrix(cov(d))
    all(isfinite, Σ) || throw(MolewhackerGeometryError())
    L = cholesky(Symmetric(Σ)).L
    A = L \ view(J, 1:m, :)
    covariance_J = view(J, m+1:size(J, 1), :)
    all(iszero, covariance_J) && return A' * A
    B = map(eachcol(covariance_J)) do v
        vec(L \ reshape(v, m, m) / L')
    end
    C = reduce(hcat, B)
    return A' * A + (C' * C) ./ 2
end

function _mw_pullback(d::Union{Distributions.Product,Distributions.ProductDistribution,NamedTupleDist}, J)
    factors = _mw_factors(d)
    ends = cumsum(map(_mw_nparams, factors))
    return mapreduce(+, eachindex(factors)) do i
        firstrow = i == firstindex(factors) ? 1 : ends[i - 1] + 1
        _mw_pullback(factors[i], view(J, firstrow:ends[i], :))
    end
end

function _mw_fisher(f, x, ad, d)
    # Share the full model's AD pass across product leaves. Their Fisher terms add.
    _, J = with_jacobian(_mw_parameters ∘ f, x, AbstractMatrix, ad)
    return _mw_pullback(d, J)
end

function _mw_precision(f, x::AbstractVector{T}, ad) where T
    all(isfinite, x) || throw(MolewhackerGeometryError())
    G = Matrix{T}(_mw_fisher(f, x, ad, f(x)))
    all(isfinite, G) || throw(MolewhackerGeometryError())
    return Matrix(Symmetric(G)) + I
end

function _mw_local_precision(f, x, ad)
    try
        P = _mw_precision(f, x, ad)
        cholesky(Symmetric(P))
        return P
    catch err
        if err isa Union{MolewhackerGeometryError,PosDefException,SingularException}
            return nothing
        end
        rethrow()
    end
end
