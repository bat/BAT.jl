# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.PolarShellDistribution{T<:Real} <: Distributions.Distribution{Multivariate,Continuous}

*Experimental feature, not yet part of stable public API.*

A doughnut-like distribution in two dimensions, in `[r, phi]` polar
coordinates.

The distribution results from transforming the radial component of a base
distribution the transport from a standard normal to a given radial
distribution.

Constructor:
```
    PolarShellDistribution(
        base_dist = MvNormal(Diagonal([1,1])),
        radial_dist = LogNormal(0, 1)
    )
```
"""
struct PolarShellDistribution{
    DB<:Distributions.Distribution{Multivariate,Continuous},
    DR<:Distributions.Distribution{Univariate,Continuous},
    M<:AbstractMeasure
} <: Distributions.Distribution{Multivariate,Continuous}
    _base_dist::DB
    _radial_dist::DR
    _m::M
end

function PolarShellDistribution(
    base_dist::Distributions.Distribution{Multivariate,Continuous} = MvNormal(Diagonal([1,1])),
    radial_dist::Distributions.Distribution{Univariate,Continuous} = Rayleigh()
)
    base_measure = batmeasure(base_dist)
    r_transform = BAT.DistributionTransform(radial_dist, truncated(Normal(), 0, Inf))
    shell_transform = ShellRTransform(r_transform)
    full_transform = ffcomp(shell_transform, _cart_to_polar)
    m = pushfwd(full_transform, base_measure)

    return PolarShellDistribution(base_dist, radial_dist, m)
end


Distributions.insupport(d::PolarShellDistribution, x::AbstractVector) = length(d) == length(x)

Base.eltype(::Type{PolarShellDistribution{DB}}) where DB = eltype(DB)

Base.length(d::PolarShellDistribution) = length(d._base_dist)

# Not applicable:
# # StatsBase.params(d::PolarShellDistribution)
# # @inline Distributions.partype(d::PolarShellDistribution)

# Can't compute efficiently:
# Statistics.mean(d::PolarShellDistribution)
# Statistics.var(d::PolarShellDistribution)
# Statistics.cov(d::PolarShellDistribution)
# StatsBase.mode(d::PolarShellDistribution)
# StatsBase.modes(d::PolarShellDistribution)


function Distributions._logpdf(d::PolarShellDistribution, x::AbstractVector)
    logdensityof(d._m, x)
end

function Distributions._pdf(d::PolarShellDistribution, x::AbstractVector)
    densityof(d._m, x)
end


function Distributions._rand!(rng::AbstractRNG, d::PolarShellDistribution, A::AbstractVector{T}) where {T<:Real}
    A .= rand(rng, d._m)
end

function Distributions._rand!(rng::AbstractRNG, d::PolarShellDistribution, A::AbstractMatrix{T}) where {T<:Real}
    A .= flatview(rand(rng, d._m^Base.tail(size(A))))
end


std_dist_from(src_d::PolarShellDistribution) = StandardMvNormal(length(src_d))

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::PolarShellDistribution, src_v::AbstractVector{<:Real})
    @argcheck length(trg_d) == length(src_d)
    apply_dist_trafo(trg_d, src_d._m.origin.dist, src_d._m.finv(src_v))
end

std_dist_to(trg_d::PolarShellDistribution) = StandardMvNormal(length(trg_d))

function apply_dist_trafo(trg_d::PolarShellDistribution, src_d::StandardMvNormal, src_v::AbstractVector{<:Real})
    @argcheck length(trg_d) == length(src_d)
    trg_d._m.f(apply_dist_trafo(trg_d._m.origin.dist, src_d, src_v))
end



function _cart_to_polar(x)
    x_1, x_2 = x
    phi = atan(x_2, x_1)
    r = sqrt(x_1^2 + x_2^2)
    return [r, phi]
end

InverseFunctions.inverse(::typeof(_cart_to_polar)) = _polar_to_cart

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(_cart_to_polar), x)
    r_phi = _cart_to_polar(x)
    r, _ = r_phi
    ladj = - log(r)
    return r_phi, ladj
end


function _polar_to_cart(r_phi)
    r, phi = r_phi
    x = r * cos(phi)
    y = r * sin(phi)
    
    return [x, y]
end

InverseFunctions.inverse(::typeof(_polar_to_cart)) = _cart_to_polar

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(_polar_to_cart), r_phi)
    r, _ = r_phi
    ladj = log(r)
    x_y = _polar_to_cart(r_phi)
    return x_y, ladj
end



struct ShellRTransform{F}
    r_transform::F
end

function (f::ShellRTransform)(r_phi)
    r, phi = r_phi
    
    return [f.r_transform(r), phi]
end

InverseFunctions.inverse(f::ShellRTransform) = ShellRTransform(inverse(f.r_transform))

function ChangesOfVariables.with_logabsdet_jacobian(f::ShellRTransform, r_phi)
    r, phi = r_phi
    new_r, ladj = with_logabsdet_jacobian(f.r_transform, r)
    return [new_r, phi], ladj
end
