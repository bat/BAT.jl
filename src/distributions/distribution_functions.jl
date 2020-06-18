# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _check_rand_compat(s::Sampleable{Multivariate}, A::Union{AbstractVector,AbstractMatrix})
    size(A, 1) == length(s) || throw(DimensionMismatch("Output size inconsistent with sample length."))
    nothing
end



@doc doc"""
    bat_sampler(d::Distribution)

*BAT-internal, not part of stable public API.*

Tries to return a BAT-compatible sampler for Distribution d. A sampler is
BAT-compatible if it supports random number generation using an arbitrary
`AbstractRNG`:

    rand(rng::AbstractRNG, s::SamplerType)
    rand!(rng::AbstractRNG, s::SamplerType, x::AbstractArray)

If no specific method of `bat_sampler` is defined for the type of `d`, it will
default to `sampler(d)`, which may or may not return a BAT-compatible
sampler.
"""
function bat_sampler end

# ToDo/Decision: Rename? Replace with `bat_default` mechanism?
bat_sampler(d::Distribution) = Distributions.sampler(d)



@doc doc"""
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

issymmetric_around_origin(d::Distributions.GenericMvTDist) = d.zeromean


const PosDefMatLike = Union{AbstractMatrix{T},AbstractPDMat{T},Cholesky{T}} where {T<:Real}


function cov2pdmat(::Type{T}, Σ::Matrix{<:Real}) where {T<:Real}
    Σ_conv = convert(Matrix{T}, Σ)
    PDMat(cholesky(Positive, Hermitian(Σ_conv)))
end

cov2pdmat(::Type{T}, Σ::PDMat{T}) where {T<:Real} = Σ
cov2pdmat(::Type{T}, Σ::AbstractPDMat) where {T<:Real} = cov2pdmat(T, Matrix(Σ))

cov2pdmat(::Type{T}, Σ::Cholesky{T}) where {T<:Real} = PDMat(Σ)
cov2pdmat(::Type{T}, Σ::Cholesky) where {T<:Real} = cov2pdmat(T, Matrix(Σ))



function get_cov end

get_cov(d::Distributions.GenericMvTDist) = d.Σ


function set_cov end

function set_cov(d::Distributions.GenericMvTDist{T,Cov}, Σ::PosDefMatLike) where {T,Cov<:PDMat{T}}
    Σ_conv = cov2pdmat(T, Σ)
    Distributions.GenericMvTDist(d.df, deepcopy(d.μ), Σ_conv)
end
