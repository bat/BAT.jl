# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _check_rand_compat(s::Sampleable{Multivariate}, A::Union{AbstractVector,AbstractMatrix})
    size(A, 1) == length(s) || throw(DimensionMismatch("Output size inconsistent with sample length."))
    nothing
end


doc"""
    bat_sampler(d::Distribution)

Tries to return a BAT-compatible sampler for Distribution d. A sampler is
BAT-compatible it it supports random number generation using an arbitrary
`AbstractRNG`:

    rand(rng::AbstractRNG, s::SamplerType)
    rand!(rng::AbstractRNG, s::SamplerType, x::AbstractArray)

If no specific method of `bat_sampler` is defined for the type of `d`, it will
default to `sampler(d)`, which may or may not return a BAT-compatible
sampler.
"""
function bat_sampler end
export bat_sampler

bat_sampler(d::Distribution) = Distributions.sampler(d)


doc"""
    issymmetric_around_origin(d::Distribution)

Returns `true` (resp. `false`) if the Distribution is symmetric (resp.
non-symmetric) around the origin.
"""
function issymmetric_around_origin end
export issymmetric_around_origin


issymmetric_around_origin(d::Normal) = d.μ ≈ 0

issymmetric_around_origin(d::Gamma) = false

issymmetric_around_origin(d::Chisq) = false

issymmetric_around_origin(d::MvNormal) = _iszero(d.μ)

issymmetric_around_origin(d::Distributions.GenericMvTDist) = d.zeromean



get_cov(d::Distributions.GenericMvTDist) = d.Σ
set_cov!(d::Distributions.GenericMvTDist{T,M}, Σ::M) where {T,M} = Distributions.GenericMvTDist{T,M}(d.df, d.dim, d.zeromean, d.μ, Σ)
set_cov!(d::Distributions.GenericMvTDist{T,M}, Σ::AbstractMatrix{<:Real}) where {T,M<:PDMat} = Distributions.GenericMvTDist{T,M}(d.df, d.dim, d.zeromean, d.μ, PDMat(Σ))


# Generic rand and rand! implementations similar to those in Distributions,
# but with an rng argument:

export BATSampler

abstract type BATSampler{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end


@inline Base.rand(s::BATSampler, args...) = rand(Base.GLOBAL_RNG, s, args...)
@inline Base.rand!(s::BATSampler, args...) = rand!(Base.GLOBAL_RNG, s, args...)

# To avoid ambiguity with Distributions:
@inline Base.rand(s::BATSampler, dims::Int...) = rand(Base.GLOBAL_RNG, s, dims...)
@inline Base.rand!(s::BATSampler, A::AbstractVector) = rand!(Base.GLOBAL_RNG, s, A)
@inline Base.rand!(s::BATSampler, A::AbstractMatrix) = rand!(Base.GLOBAL_RNG, s, A)


function Base.rand!(rng::AbstractRNG, s::BATSampler{Univariate}, A::AbstractArray)
    @inbounds @simd for i in 1:length(A)
        A[i] = rand(rng, s)
    end
    return A
end

Base.rand(rng::AbstractRNG, s::BATSampler{Univariate}, dims::Dims) =
    rand!(rng, s, Array{eltype(s)}(dims))

Base.rand(rng::AbstractRNG, s::BATSampler{Univariate}, dims::Int...) =
    rand!(s, Array{eltype(s)}(dims))


Base.rand(rng::AbstractRNG, s::BATSampler{Multivariate}) =
    rand!(rng, s, Vector{eltype(s)}(length(s)))

Base.rand(rng::AbstractRNG, s::BATSampler{Multivariate}, n::Integer) =
    rand!(rng, s, Matrix{eltype(s)}(length(s), n))

function Base.rand!(rng::AbstractRNG, s::BATSampler{Multivariate}, A::AbstractMatrix)
    _check_rand_compat(s, A)
    @inbounds for i = 1:size(A,2)
        rand!(rng, s, view(A,:,i))
    end
    return A
end

Base.rand(rng::AbstractRNG, s::BATSampler{Multivariate}, n::Int) = rand!(rng, s, Matrix{eltype(s)}(length(s), n))



function _rand_gamma_mt(rng::AbstractRNG, ::Type{T}, shape::Real) where {T<:Real}
    (shape <= 0) && throw(ArgumentError("Require shape > 0, got $shape"))

    α = T(shape)

    if (α <= 1)
        return _rand_gamma_mt(rng, T, α + 1) * rand(rng, T)^(1/α)
    else
        k = T(3)
        d = α - 1/k;
        c = 1 / (k * √d);  # == 1 / √(k^2 * α - k)

        while true
            x = randn(rng, T)
            cx1 = c*x + 1
            if (0 < cx1)  # -1/c < x
                h_x = d * cx1^3  # hx(x) = d * (1 + c*x)^3
                u = rand(rng, T);

                v = cx1^3;
                dv = d*v  # == h(x) = d * (1 + c*x)^3
                if (u > 0)
                    (u < 1 - T(0.0331) * x^4) && return dv
                    (log(u) < x^2/2 + (d - dv + d * log(v))) && return dv
                end
            end
        end
    end
end

export BATGammaMTSampler

struct BATGammaMTSampler{T} <: BATSampler{Univariate,Continuous}
    shape::T
    scale::T
end

BATGammaMTSampler(d::Gamma) = BATGammaMTSampler(shape(d), scale(d))

Base.eltype(s::BATGammaMTSampler{T}) where {T} = T

Base.rand(rng::AbstractRNG, s::BATGammaMTSampler) = s.scale * _rand_gamma_mt(rng, float(typeof(s.shape)), s.shape)

bat_sampler(d::Gamma) = BATGammaMTSampler(d)


export BATChisqSampler

struct BATChisqSampler{T} <: BATSampler{Univariate,Continuous}
    gamma_sampler::T
end

function BATChisqSampler(d::Chisq)
    shape = dof(d) / 2
    scale = typeof(shape)(2)
    BATChisqSampler(BATGammaMTSampler(shape, scale))
end

Base.eltype(s::BATChisqSampler) = eltype(s.gamma_sampler)

Base.rand(rng::AbstractRNG, s::BATChisqSampler) = rand(rng, s.gamma_sampler)

bat_sampler(d::Chisq) = BATChisqSampler(d)



export BATMvTDistSampler

struct BATMvTDistSampler{T<:Distributions.GenericMvTDist} <: BATSampler{Multivariate,Continuous}
    d::T
end


Base.length(s::BATMvTDistSampler) = length(s.d)

Base.eltype(s::BATMvTDistSampler) = eltype(s.d.Σ)



_tdist_scaling_factor(rng::AbstractRNG, x::AbstractVector, df::Real) = sqrt(df / rand(rng, bat_sampler(Chisq(df))))

function _tdist_scaling_factor(rng::AbstractRNG, x::AbstractMatrix, df::Real)
    scaling_factor = similar(x, 1, size(x, 2)) # TODO: Avoid memory allocation
    rand!(rng, bat_sampler(Chisq(df)), scaling_factor)
    scaling_factor .= sqrt.(df ./ scaling_factor)
end


function Base.rand!(rng::AbstractRNG, s::BATMvTDistSampler, x::StridedVecOrMat{T}) where {T<:Real}
    d = s.d

    scaling_factor = _tdist_scaling_factor(rng, x, d.df)
    unwhiten!(d.Σ, randn!(rng, x))  # TODO: Avoid memory allocation (caused by Σ.chol[:UL] in unwhiten! implementation)
    broadcast!(muladd, x, x, scaling_factor, d.μ)
    x
end



bat_sampler(d::Distributions.GenericMvTDist) = BATMvTDistSampler(d)
