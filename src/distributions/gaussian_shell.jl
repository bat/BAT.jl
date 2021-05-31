# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using SpecialFunctions
using Random123 
import AdaptiveRejectionSampling

"""
    struct BAT.GaussianShell <: Distribution{Multivariate,Continuous}

*Experimental feature, not part of stable public API.*

Gaussian Shell (see
[Caldwell et al.](https://arxiv.org/abs/1808.08051) for definition).

Constructors:

* ```BAT.GaussianShell(; r::Real=5, w::Real=2, n::Integer=2)```

Fields:

$(TYPEDFIELDS)

!!! note

    Fields `c` and `lognorm` are considered internal and subject to
    change without deprecation.
"""
struct GaussianShell{T<:Real, V<:AbstractVector{<:Real}, F<:AbstractFloat} <: Distribution{Multivariate,Continuous}
    "The radius of the Gaussian shell distribution."
    r::T

    "Variance of the Gaussian shell distribution"
    w::T

    "Number of dimensions"
    n::Int

    c::V

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
    GaussianShell(r, w, n, c, lognorm)
end

function _find_gs_search_range(ndims::Integer, r0::Real, w::Real)
    f(r) = gs_pdf_r(r, ndims, r0, w)
    o = AdaptiveRejectionSampling.Objective(f)

    r_grid = Array(range(maximum([r0/2, r0-w]), stop=sqrt(2*ndims)*w+r0/(w^2) + sqrt(ndims)*r0/w, length=10^6))
    y_grid = f.(r_grid)
    y_grads = o.grad.(r_grid)

    # Both sides of a concave peak
    idx1 = argmax(y_grads)
    idx2 = argmin(y_grads)

    return (r_grid[idx1], r_grid[idx2]+sqrt(ndims)*r0)
end

nball_surf_area(r, ndims) = 2 * π^(ndims/2) / gamma(ndims/2) * r^(ndims-1)
gs_pdf_r(r, ndims, r0, w) = nball_surf_area(r, ndims) * 1/(2π * w^2) * exp(-(abs(r) - r0)^2 / (2 * w^2))

function gauss_shell_radial_samples(rng::AbstractRNG, ndims::Integer, r0::Real, w::Real, n::Integer)
    f(r) = gs_pdf_r(r, ndims, r0, w)
    result = Vector{Float64}()

    # ToDo: AdaptiveRejectionSampling sometimes fails sporadically here,
    # find a fix to replace this workaround:
    ntries::Int = 0
    while length(result) < n
        ntries += 1
        try
            sampler = AdaptiveRejectionSampling.RejectionSampler(f, (10^-10, Inf), search_range=_find_gs_search_range(ndims, r0, w))
            smpls = AdaptiveRejectionSampling.run_sampler!(rng, sampler, n)
            append!(result, smpls)
        catch err
            if !(err isa AssertionError || err isa DomainError) || ntries > 4
                rethrow()
            end
        end
    end

    result
end

function rand_nball_surf_samples!(rng::AbstractRNG, X::AbstractMatrix)
    randn!(rng, X)
    X ./= sqrt.(sum(X .* X, dims = 1))
    return X
end

Distributions.rand(rng::AbstractRNG, d::GaussianShell) = vec(rand(rng, d, 1))

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
    cov(nestedview(rand(bat_determ_rng(), bat_sampler(d), 10^5)))
end

Base.length(d::GaussianShell) = length(d.c)

Base.eltype(d::GaussianShell) = eltype(d.c)

Distributions.params(d::GaussianShell) = (d.r, d.w, d.c)
