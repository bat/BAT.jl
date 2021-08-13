# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct RenormalizedDensity <: AbstractDensity

Constructors:

* ```$(FUNCTIONNAME)(density::AbstractDensity, logrenormf::Real)```

A renormalized density derived from `density`, with

```julia
logdensityof(RenormalizedDensity(density, logrenormf))(v) ==
    logdensityof(density)(v) + logrenormf
```
"""
struct RenormalizedDensity{D<:AbstractDensity,T<:Real} <: AbstractDensity
    density::D
    logrenormf::T
end

Base.parent(density::RenormalizedDensity) = density.density

Base.:(==)(a::RenormalizedDensity, b::RenormalizedDensity) = a.density == b.density && a.logrenormf == b.logrenormf

var_bounds(density::RenormalizedDensity) = var_bounds(parent(density))

ValueShapes.varshape(density::RenormalizedDensity) = varshape(parent(density))

ValueShapes.unshaped(density::RenormalizedDensity) = RenormalizedDensity(unshaped(density.density), density.logrenormf)

(shape::AbstractValueShape)(density::RenormalizedDensity) = RenormalizedDensity(shape(density.density), density.logrenormf)


function Base.show(io::IO, d::RenormalizedDensity)
    print(io, Base.typename(typeof(d)).name, "(")
    show(io, d.density)
    print(io, ", ")
    show(io, d.logrenormf)
    print(io, ")")
end


function DensityInterface.logdensityof(density::RenormalizedDensity, v::Any)
    parent_logd = logdensityof(parent(density),v)
    R = float(typeof(parent_logd))
    convert(R, parent_logd + density.logrenormf)
end

function checked_logdensityof(density::RenormalizedDensity, v::Any)
    parent_logd = checked_logdensityof(parent(density),v)
    R = float(typeof(parent_logd))
    convert(R, parent_logd + density.logrenormf)
end



Distributions.sampler(density::RenormalizedDensity) = Distributions.sampler(parent(density))
bat_sampler(density::RenormalizedDensity) = bat_sampler(parent(density))

Statistics.cov(density::RenormalizedDensity) = cov(parent(density))


"""
    BAT.renormalize_density(density::AbstractDensity, logrenormf::Real)::AbstractDensity

*Experimental feature, not part of stable public API.*

Renormalies `density` with the logarithmic renormalization factor, so that

```julia
logdensityof(renormalize_density(density, logrenormf))(v) ==
    logdensityof(density)(v) + logrenormf
```
"""
function renormalize_density end
export renormalize_density

renormalize_density(density::Any, logrenormf::Real) = renormalize_density(convert(AbstractDensity, density), logrenormf)

renormalize_density(density::AbstractDensity, logrenormf::Real) = RenormalizedDensity(density, logrenormf)

renormalize_density(density::RenormalizedDensity, logrenormf::Real) = renormalize_density(parent(density), density.logrenormf + logrenormf)
