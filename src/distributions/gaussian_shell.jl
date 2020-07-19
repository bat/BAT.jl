using SpecialFunctions
using Random123 
import AdaptiveRejectionSampling
#import HCubature

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
"""
struct GaussianShell{T<:Real, V<:AbstractVector, F<:Float64} <: ContinuousMultivariateDistribution
    r::T
    w::T
    c::V
    n::Int
    lognorm::F
end

function GaussianShell(;r::Real=5, w::Real=2, n::Integer=2)
    c = zeros(n)
    r,w = promote(r, w)
    f(x) = x^(n-1)*exp(-(x - r)^2 / (2*w^2))
    lognorm = log((sqrt(2)*π^((n-1)/2)) / (gamma(n/2)*w)*QuadGK.quadgk(f, 0, r+w*20)[1])
    GaussianShell(r, w, c, n, lognorm)
end

nball_surf_area(r, ndims) = 2 * π^(ndims/2) / gamma(ndims/2) * r^(ndims-1)
gs_pdf_r(r, ndims, r0, w) = nball_surf_area(r, ndims) * 1/(2π * w^2) * exp(-(abs(r) - r0)^2 / (2 * w^2))

function gauss_shell_radial_samples(rng::AbstractRNG, ndims::Integer, r0::Real, w::Real, n::Integer)
    f(r) = gs_pdf_r(r, ndims, r0, w)
    sampler = AdaptiveRejectionSampling.RejectionSampler(f, (10^-10, Inf))
    AdaptiveRejectionSampling.run_sampler!(rng, sampler, n)
end

function rand_nball_surf_samples(rng::AbstractRNG, ndims::Integer, n::Integer)
    A = randn(rng, ndims, n)
    A ./= sqrt.(sum(A .* A, dims = 1))
end

function Distributions.rand(rng::AbstractRNG, d::GaussianShell, n::Int)
    Distributions._rand!(rng, d, Matrix{eltype(d)}(undef, length(d), n), n)
end

function Distributions._rand!(rng::AbstractRNG, d::GaussianShell, X::AbstractMatrix, n::Int)
    X = rand_nball_surf_samples(rng, d.n, n)
    R = gauss_shell_radial_samples(rng, d.n, d.r, d.w, n)
    X .* R'
end

function Distributions._logpdf(d::GaussianShell, x::AbstractArray)
    pdf_normalization = -log(sqrt(2*π*(d.w)^2))
    integral_result = -(sqrt(sum((x .- d.c).^2)) - d.r)^2 / (2*d.w^2)
    result = integral_result + pdf_normalization - d.lognorm
    return result
end

Base.length(d::GaussianShell) = length(d.c)

Base.eltype(d::GaussianShell) = eltype(d.c)

Distributions.params(d::GaussianShell) = (d.r, d.w, d.c)
