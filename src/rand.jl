# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Compat
using Distributions


export DistForRNG

"""
    DistForRNG{T <: Distribution}

Wrapper type to add support for random number generation using an arbitrary
`AbstractRNG` to selected subtypes of `Distributions.Distribution`.

Examples:

    rand(DistForRNG(Gamma(4, 2)), MersenneTwister(7002), Float64)

    issymmetric_around_origin(DistForRNG(Normal())) == true
    issymmetric_around_origin(DistForRNG(Normal(2,2))) == false
"""
immutable DistForRNG{T <: Distribution}
    d::T
end


export issymmetric_around_origin

"""
    issymmetric_around_origin(d::DistForRNG)::Bool

Returns `true` (resp. `false`) if the Distribution is symmetric (resp.
non-symmetric) around the origin.
"""
function issymmetric_around_origin end



"""
    Base.rand{T}(d::DistForRNG{Gamma}, ::Type{T} = Float64)
    Base.rand(d::DistForRNG{Gamma}, r::AbstractRNG)
    Base.rand{T<:AbstractFloat}(d::DistForRNG, r::AbstractRNG, ::Type{T})
"""
Base.rand{T}(d::DistForRNG, ::Type{T} = Float64) = rand(d, Base.GLOBAL_RNG, T)
Base.rand(d::DistForRNG, r::AbstractRNG) = rand(d, r, Float64)


Base.rand{D<:Normal,T<:AbstractFloat}(d::DistForRNG{D}, r::AbstractRNG, ::Type{T}) = d.d.σ * randn(r, T) + d.d.μ
issymmetric_around_origin{D<:Normal}(d::DistForRNG{D}) = d.d.μ ≈ 0

Base.rand{D<:Gamma,T<:AbstractFloat}(d::DistForRNG{D}, r::AbstractRNG, ::Type{T}) = rand_gamma(r, T, d.d.α, d.d.θ)
issymmetric_around_origin{D<:Gamma,}(d::DistForRNG{D}) = false

Base.rand{D<:Chisq,T<:AbstractFloat}(d::DistForRNG{D}, r::AbstractRNG, ::Type{T}) = rand_chisq(r, T, d.d.ν)
issymmetric_around_origin{D<:Chisq,}(d::DistForRNG{D}) = false



rand_gamma{T}(rng::AbstractRNG, ::Type{T}, shape::Real, scale::Real) =
    rand_gamma(rng, T, shape) * scale

function rand_gamma{T<:Real}(rng::AbstractRNG, ::Type{T}, shape::Real)
    (shape <= 0) && throw(ArgumentError("Require shape > 0, got $shape"))

    α = T(shape)

    if (α <= 1)
        return rand_gamma(rng, T, α + 1) * rand(rng, T)^(1/α)
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


rand_chisq{T}(rng::AbstractRNG, ::Type{T}, dof::Real) = rand_gamma(rng, T, dof / 2, 2)
