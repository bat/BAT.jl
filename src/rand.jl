# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Compat
using Distributions, PDMats


function _check_rand_compat(s::Sampleable{Multivariate}, A::Union{AbstractVector,AbstractMatrix})
    size(A, 1) == length(s) || throw(DimensionMismatch("Output size inconsistent with sample length."))
    nothing
end


export bat_sampler

"""
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

bat_sampler(d::Distribution) = Distributions.sampler(d)



export issymmetric_at_origin

"""
    issymmetric_at_origin(d::DistForRNG)

Returns `true` (resp. `false`) if the Distribution is symmetric (resp.
non-symmetric) around the origin.
"""
function issymmetric_at_origin end


issymmetric_at_origin(d::Normal) = d.μ ≈ 0

issymmetric_at_origin(d::Gamma) = false

issymmetric_at_origin(d::Chisq) = false

issymmetric_at_origin(d::MvNormal) = all(x -> x == 0, mv.μ)

issymmetric_at_origin(d::Distributions.GenericMvTDist) = d.zeromean



# Generic rand and rand! implementations similar to those in Distributions,
# but with an rng argument:

export BATSampler

@compat abstract type BATSampler{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end


@inline Base.rand(s::BATSampler, args...) = rand(Base.GLOBAL_RNG, s, args...)
@inline Base.rand!(s::BATSampler, args...) = rand!(Base.GLOBAL_RNG, s, args...)

# To avoid ambiguity with Distributions:
@inline Base.rand(s::BATSampler, dims::Int64...) = rand(Base.GLOBAL_RNG, s, dims...)
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
    rand!(s, Vector{eltype(s)}(length(s)))

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



function _rand_gamma_mt{T<:Real}(rng::AbstractRNG, ::Type{T}, shape::Real)
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

immutable BATGammaMTSampler{T} <: BATSampler{Univariate,Continuous}
    shape::T
    scale::T
end

BATGammaMTSampler(d::Gamma) = BATGammaMTSampler(shape(d), scale(d))

Base.eltype{T}(s::BATGammaMTSampler{T}) = T

Base.rand(rng::AbstractRNG, s::BATGammaMTSampler) = s.scale * _rand_gamma_mt(rng, float(typeof(s.shape)), s.shape)

bat_sampler(d::Gamma) = BATGammaMTSampler(d)


export BATChisqSampler

immutable BATChisqSampler{T} <: BATSampler{Univariate,Continuous}
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

immutable BATMvTDistSampler{T<:Distributions.GenericMvTDist} <: BATSampler{Multivariate,Continuous}
    d::T
end


Base.length(s::BATMvTDistSampler) = length(s.d)

Base.eltype(s::BATMvTDistSampler) = eltype(s.d.Σ)

# Based on implementation of Distributions._rand!{T<:Real}(d::GenericMvTDist, x::AbstractVector{T}):
function Base.rand!{T<:Real}(rng::AbstractRNG, s::BATMvTDistSampler, x::AbstractVector{T})
    d = s.d
    chisqd = Chisq(d.df)
    y = sqrt(rand(rng, bat_sampler(chisqd))/(d.df))
    unwhiten!(d.Σ, randn!(rng, x))
    broadcast!(/, x, x, y)
    if !d.zeromean
        broadcast!(+, x, x, d.μ)
    end
    x
end

bat_sampler(d::Distributions.GenericMvTDist) = BATMvTDistSampler(d)
