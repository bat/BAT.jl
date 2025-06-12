# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.PushForwardDistribution{VF,SP} <: Distributions.Distribution{VF,SP}

*Experimental feature, not yet part of stable public API.*

A doughnut-like distribution in two dimensions, in `[r, phi]` polar
coordinates.

The distribution results from transforming the radial component of a base
distribution the transport from a standard normal to a given radial
distribution.

Constructor:
```
PushForwardDistribution(f, origin::Distribution)
```
"""
struct PushForwardDistribution{
    VF <: VariateForm, SP<:ValueSupport, F, FI, D<:Distribution, SZ<:Dims, SHP<:ValueShape
} <: Distributions.Distribution{VF,SP}
    f::F
    finv::FI
    origin::D
    _size::SZ
    _vs::SHP
end

function PushForwardDistribution(f::F, d::D) where {F,DVF,SP,D<:Distribution{DVF,SP}}
    finv = inverse(f)
    shp = pushfwd_varshape(f, d)
    VF = Univariate#!!!!!!!
    return PushForwardDistribution{VF,SP,F,typeof(finv),D,typeof(SHP)}(f, finv, d, shp)
end



Distributions.insupport(d::PushForwardDistribution, x::AbstractVector) = insupport(d.origin, d.finv(x))

# ToDo: Useful/sensible?
#Base.eltype(::Type{PushForwardDistribution{F,FI,D}}) where {F,FI,D} = eltype(D)

Base.size(d::PushForwardDistribution) = d._size
Base.length(d::PushForwardDistribution) = prod(size(d))


# Unclear semantics for PushForwardDistribution:
# # StatsBase.params(d::PushForwardDistribution)
# # @inline Distributions.partype(d::PushForwardDistribution)

# Can't compute efficiently in the general case:
# Statistics.mean(d::PushForwardDistribution)
# Statistics.var(d::PushForwardDistribution)
# Statistics.cov(d::PushForwardDistribution)
# StatsBase.mode(d::PushForwardDistribution)
# StatsBase.modes(d::PushForwardDistribution)


function Distributions._logpdf(d::PushForwardDistribution, x::AbstractVector)
    logdensityof(d._m, x)
end

function Distributions._pdf(d::PushForwardDistribution, x::AbstractVector)
    densityof(d._m, x)
end


function Distributions._rand!(rng::AbstractRNG, d::PushForwardDistribution, A::AbstractVector{T}) where {T<:Real}
    A .= rand(rng, d._m)
end

function Distributions._rand!(rng::AbstractRNG, d::PushForwardDistribution, A::AbstractMatrix{T}) where {T<:Real}
    A .= flatview(rand(rng, d._m^Base.tail(size(A))))
end


ValueShapes.varshape(d::PushForwardDistribution) = d._vs


std_dist_from(src_d::PushForwardDistribution) = std_dist_from(src_d.origin)

function apply_dist_trafo(trg_d::StandardMvNormal, src_d::PushForwardDistribution, src_v::AbstractVector{<:Real})
    @argcheck length(trg_d) == length(src_d)
    apply_dist_trafo(trg_d, src_d._m.origin.dist, src_d._m.finv(src_v))
end

std_dist_to(trg_d::PushForwardDistribution) = std_dist_to(trg_d.origin)

function apply_dist_trafo(trg_d::PushForwardDistribution, src_d::StandardMvNormal, src_v::AbstractVector{<:Real})
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
