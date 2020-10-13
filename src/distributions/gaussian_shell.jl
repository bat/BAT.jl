# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using SpecialFunctions
using Random123 
import AdaptiveRejectionSampling

"""
    BAT.GaussianShell([r=5, w=2, n=2])

Gaussian Shell (Caldwell et al.)[https://arxiv.org/abs/1808.08051].

# Arguments
- `r::Real`: The radius of the Gaussian shell distribution.
- `w::Real`: Variance of the Gaussian shell distribution.
- `n::Int`: Number of dimensions.

Constructors:
```julia
BAT.GaussianShell(r::Real, w::Real, c::Vector, n::Int)
```
"""
struct GaussianShell{T<:Real, V<:AbstractVector{<:Real}, F<:AbstractFloat} <: ContinuousMultivariateDistribution
    r::T
    w::T
    c::V
    n::Int
    lognorm::F
end

function GaussianShell(;r::Real=5, w::Real=2, n::Integer=2)
    c = zeros(n)
    r,w = promote(r, w)
    radial_integral(ρ) = ρ^(n-1)*exp(-(ρ - r)^2 / (2*w^2))
    density_norm = sqrt(2*π*w^2)
    # Normalization for coordinate transform
    radial_norm = (sqrt(2)*π^((n-1)/2)) / (gamma(n/2)*w)*QuadGK.quadgk(radial_integral, 0, r+w*20)[1]
    lognorm = log(density_norm) + log(radial_norm)
    GaussianShell(r, w, c, n, lognorm)
end

nball_surf_area(r, ndims) = 2 * π^(ndims/2) / gamma(ndims/2) * r^(ndims-1)
gs_pdf_r(r, ndims, r0, w) = nball_surf_area(r, ndims) * 1/(2π * w^2) * exp(-(abs(r) - r0)^2 / (2 * w^2))

function gauss_shell_radial_samples(rng::AbstractRNG, ndims::Integer, r0::Real, w::Real, n::Integer)
    f(r) = gs_pdf_r(r, ndims, r0, w)
    sampler = AdaptiveRejectionSampling.RejectionSampler(f, (10^-10, Inf))
    AdaptiveRejectionSampling.run_sampler!(rng, sampler, n)
end

function rand_nball_surf_samples!(rng::AbstractRNG, X::AbstractMatrix)
    randn!(rng, X)
    X ./= sqrt.(sum(X .* X, dims = 1))
    return X
end

function Distributions.rand(rng::AbstractRNG, d::GaussianShell, n::Int)
    X = Matrix{eltype(d)}(undef, length(d), n)
    Distributions._rand!(rng, d, X)
end

function Distributions._rand!(rng::AbstractRNG, d::GaussianShell, X::AbstractMatrix)
    rand_nball_surf_samples!(rng, X)
    R = gauss_shell_radial_samples(rng, d.n, d.r, d.w, size(X,2))
    X .*= R'
    return X
end

function Distributions._logpdf(d::GaussianShell, x::AbstractArray)
    integral_result = -(sqrt(sum((x .- d.c).^2)) - d.r)^2 / (2*d.w^2)
    result = integral_result - d.lognorm
    return result
end

function Statistics.cov(d::GaussianShell)
    cov(nestedview(rand(bat_determ_rng(), sampler(d), 10^5)))
end

Base.length(d::GaussianShell) = length(d.c)

Base.eltype(d::GaussianShell) = eltype(d.c)

Distributions.params(d::GaussianShell) = (d.r, d.w, d.c)
