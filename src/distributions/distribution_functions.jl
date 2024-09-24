# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function eff_totalndof end

eff_totalndof(d::Distribution{Univariate}) = length(d)
eff_totalndof(d::Distribution{Multivariate}) = length(d)
eff_totalndof(d::Distribution{Matrixvariate}) = length(d)

eff_totalndof(d::ConstValueDist) = 0

# ToDo: Find more generic way of handling this:
eff_totalndof(d::NamedTupleDist) = sum(map(eff_totalndof, values(d)))
eff_totalndof(d::ValueShapes.UnshapedNTD) = eff_totalndof(d.shaped)
eff_totalndof(d::ReshapedDist) = eff_totalndof(unshaped(d))



function _check_rand_compat(s::Sampleable{Multivariate}, A::Union{AbstractVector,AbstractMatrix})
    size(A, 1) == length(s) || throw(DimensionMismatch("Output size inconsistent with sample length."))
    nothing
end


"""
    issymmetric_around_origin(d::Distribution)

*BAT-internal, not part of stable public API.*

Returns `true` (resp. `false`) if the Distribution is symmetric (resp.
non-symmetric) around the origin.
"""
function issymmetric_around_origin end


issymmetric_around_origin(d::Normal) = d.μ ≈ 0

issymmetric_around_origin(d::Gamma) = false

issymmetric_around_origin(d::Chisq) = false

issymmetric_around_origin(d::TDist) = true

issymmetric_around_origin(d::MvNormal) = iszero(d.μ)

issymmetric_around_origin(d::Distributions.GenericMvTDist) = iszero(d.μ)


const PosDefMatLike = Union{AbstractMatrix{T},AbstractPDMat{T},Cholesky{T}} where {T<:Real}


function cov2pdmat(::Type{T}, Σ::AbstractMatrix{<:Real}) where {T<:Real}
    Σ_conv = convert(Matrix{T}, Σ)
    PDMat(cholesky(Positive, Hermitian(Σ_conv)))
end

cov2pdmat(::Type{T}, Σ::PDMat{T}) where {T<:Real} = Σ
cov2pdmat(::Type{T}, Σ::AbstractPDMat) where {T<:Real} = cov2pdmat(T, Matrix(Σ))

cov2pdmat(::Type{T}, Σ::Cholesky{T}) where {T<:Real} = PDMat(Σ)
cov2pdmat(::Type{T}, Σ::Cholesky) where {T<:Real} = cov2pdmat(T, Matrix(Σ))



function get_cov end

get_cov(d::Distributions.GenericMvTDist, n::Integer) = d.Σ

get_cov(d::Distributions.TDist, n::Integer) = Diagonal(fill(var(d), n))

get_cov(d::Distributions.MultivariateDistribution, n::Integer) = cov(d)

get_cov(d::Distributions.UnivariateDistribution, n::Integer) = Diagonal(fill(var(d), n))

function set_cov end

function set_cov(d::Distributions.GenericMvTDist{T,Cov}, Σ::PosDefMatLike) where {T,Cov<:PDMat{T}}
    Σ_conv = cov2pdmat(T, Σ)
    Distributions.GenericMvTDist(d.df, deepcopy(d.μ), Σ_conv)
end

# TODO: MD, is this needed? should this stay?
set_cov(d::Distributions.MultivariateDistribution, Σ::PosDefMatLike) = typeof(d)(mean(d), Σ)
set_cov(d::Distributions.UnivariateDistribution, Σ::PosDefMatLike) = typeof(d)(mean(d), Σ[1,1])



# TODO: MD, remove. Only temporary hack
function set_cov(d::Distributions.TDist{T}, Σ::PosDefMatLike) where {T}
    Σ_conv = cov2pdmat(T, Σ)
    var = Σ_conv[1,1]

    ν = 2 * var / (var - 1)

    Distributions.TDist(ν)
end