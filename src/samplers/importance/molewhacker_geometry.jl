# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Pull back distribution Fisher information without constructing its parameter-space
# matrix. In particular, the covariance block of an m-dimensional normal would
# otherwise require O(m^4) storage.
struct MolewhackerGeometryError <: Exception end

function _mw_regular_positive(x)
    isfinite(x) && x > 0 || throw(MolewhackerGeometryError())
    return x
end

_mw_parameters(d::Union{Normal,Poisson,Exponential}) = collect(params(d))
_mw_parameters(d::MvNormal) = vcat(mean(d), vec(Matrix(cov(d))))
_mw_parameters(d::MvNormal{<:Real,<:PDiagMat}) = vcat(mean(d), d.Σ.diag)
_mw_parameters(d::MvNormal{<:Real,<:ScalMat}) = vcat(mean(d), d.Σ.value)

_mw_factors(d::Distributions.Product) = d.v
_mw_factors(d::Distributions.ProductDistribution) = vec(d.dists)
_mw_factors(d::NamedTupleDist) = values(d)

function _mw_parameters(d::Union{Distributions.Product,Distributions.ProductDistribution,NamedTupleDist})
    return reduce(vcat, map(_mw_parameters, _mw_factors(d)))
end

_mw_nparams(::Normal) = 2
_mw_nparams(::Union{Poisson,Exponential}) = 1
_mw_nparams(d::MvNormal) = length(d) + length(d)^2
_mw_nparams(d::MvNormal{<:Real,<:PDiagMat}) = 2 * length(d)
_mw_nparams(d::MvNormal{<:Real,<:ScalMat}) = length(d) + 1
function _mw_nparams(d::Union{Distributions.Product,Distributions.ProductDistribution,NamedTupleDist})
    return sum(_mw_nparams, _mw_factors(d))
end

function _mw_parameters(d)
    throw(ArgumentError("MolewhackerSampling has no Fisher geometry for $(typeof(d)). Supported families are Normal, MvNormal, Poisson, Exponential, and their products."))
end

function _mw_whitened_jacobian(d::Normal, J)
    σ = _mw_regular_positive(std(d))
    A = J ./ σ
    view(A, 2:2, :) .*= sqrt(eltype(A)(2))
    return A
end

function _mw_whitened_jacobian(d::Union{Poisson,Exponential}, J)
    m = _mw_regular_positive(mean(d))
    scale = d isa Poisson ? sqrt(m) : m
    return J ./ scale
end

function _mw_whitened_jacobian(d::MvNormal{<:Real,<:PDiagMat}, J)
    m = length(d)
    variances = _mw_regular_positive.(d.Σ.diag)
    scales = sqrt.(variances)
    Jμ, Jv = view(J, 1:m, :), view(J, m+1:2m, :)
    T = promote_type(eltype(J), eltype(variances))
    # The compact covariance parameters are variances, with information 1/(2v²).
    return vcat(Jμ ./ scales, Jv ./ variances ./ sqrt(T(2)))
end

function _mw_whitened_jacobian(d::MvNormal{<:Real,<:ScalMat}, J)
    m = length(d)
    variance = _mw_regular_positive(d.Σ.value)
    Jμ, Jv = view(J, 1:m, :), view(J, m+1:m+1, :)
    T = promote_type(eltype(J), typeof(variance))
    scale = sqrt(T(m) / T(2))
    return vcat(Jμ ./ sqrt(variance), Jv ./ variance .* scale)
end

function _mw_whitened_jacobian(d::MvNormal, J)
    m = length(d)
    Σ = Matrix(cov(d))
    all(isfinite, Σ) || throw(MolewhackerGeometryError())
    L = cholesky(Symmetric(Σ)).L
    A = L \ view(J, 1:m, :)
    covariance_J = view(J, m+1:size(J, 1), :)
    all(iszero, covariance_J) && return A
    B = map(eachcol(covariance_J)) do v
        vec(L \ reshape(v, m, m) / L')
    end
    C = reduce(hcat, B)
    C ./= sqrt(eltype(C)(2))
    return vcat(A, C)
end

function _mw_whitened_jacobian(d::Union{Distributions.Product,Distributions.ProductDistribution,NamedTupleDist}, J)
    factors = _mw_factors(d)
    ends = cumsum(map(_mw_nparams, factors))
    blocks = map(eachindex(factors)) do i
        firstrow = i == firstindex(factors) ? 1 : ends[i - 1] + 1
        _mw_whitened_jacobian(factors[i], view(J, firstrow:ends[i], :))
    end
    return reduce(vcat, blocks)
end

function _mw_pullback(d, J)
    # Stack independent information factors before forming the parameter-space Gram matrix.
    A = _mw_whitened_jacobian(d, J)
    return A' * A
end

function _mw_jacobian(f, x, ad)
    # Avoid an unused primal evaluation for default ForwardDiff. Other selectors,
    # including configured chunk sizes or tags, retain their generic AD path.
    if forward_adtype(ad) == ADSelector(ForwardDiff)
        return ForwardDiff.jacobian(f, x)
    end
    return last(with_jacobian(f, x, AbstractMatrix, ad))
end

function _mw_local_precision(f, x::AbstractVector{T}, ad) where T
    try
        all(isfinite, x) || throw(MolewhackerGeometryError())
        d = f(x)
        # Share one model Jacobian across product leaves, then add the prior.
        J = _mw_jacobian(_mw_parameters ∘ f, x, ad)
        G = Matrix{T}(_mw_pullback(d, J))
        all(isfinite, G) || throw(MolewhackerGeometryError())
        P = Matrix(Symmetric(G)) + I
        return PDMat(P, cholesky(Symmetric(P)))
    catch err
        if err isa Union{MolewhackerGeometryError,PosDefException,SingularException}
            return nothing
        end
        rethrow()
    end
end
