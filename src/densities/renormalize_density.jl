# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct Renormalized <: AbstractMeasureOrDensity

*BAT-internal, not part of stable public API.*

Constructors:

* ```$(FUNCTIONNAME)(density::AbstractMeasureOrDensity, logrenormf::Real)```

A renormalized density derived from `density`, with

```julia
logdensityof(Renormalized(density, logrenormf))(v) ==
    logdensityof(density)(v) + logrenormf
```
"""
struct Renormalized{D<:AbstractMeasureOrDensity,T<:Real} <: AbstractMeasureOrDensity
    density::D
    logrenormf::T
end

@inline DensityInterface.DensityKind(x::Renormalized) = DensityKind(x.density)

Base.parent(density::Renormalized) = density.density

Base.:(==)(a::Renormalized, b::Renormalized) = a.density == b.density && a.logrenormf == b.logrenormf

var_bounds(density::Renormalized) = var_bounds(parent(density))

ValueShapes.varshape(density::Renormalized) = varshape(parent(density))

ValueShapes.unshaped(density::Renormalized) = Renormalized(unshaped(density.density), density.logrenormf)

(shape::AbstractValueShape)(density::Renormalized) = Renormalized(shape(density.density), density.logrenormf)


function Base.show(io::IO, d::Renormalized)
    print(io, Base.typename(typeof(d)).name, "(")
    show(io, d.density)
    print(io, ", ")
    show(io, d.logrenormf)
    print(io, ")")
end


function DensityInterface.logdensityof(density::Renormalized, v::Any)
    parent_logd = logdensityof(parent(density),v)
    R = float(typeof(parent_logd))
    convert(R, parent_logd + density.logrenormf)
end

function checked_logdensityof(density::Renormalized, v::Any)
    parent_logd = checked_logdensityof(parent(density),v)
    R = float(typeof(parent_logd))
    convert(R, parent_logd + density.logrenormf)
end



Distributions.sampler(density::Renormalized) = Distributions.sampler(parent(density))
bat_sampler(density::Renormalized) = bat_sampler(parent(density))

Statistics.cov(density::Renormalized) = cov(parent(density))


"""
    BAT.renormalize_density(density::AbstractMeasureOrDensity, logrenormf::Real)::AbstractMeasureOrDensity

*Experimental feature, not part of stable public API.*

Renormalies `density` with the logarithmic renormalization factor, so that

```julia
logdensityof(renormalize_density(density, logrenormf))(v) ==
    logdensityof(density)(v) + logrenormf
```
"""
function renormalize_density end
export renormalize_density

renormalize_density(density::Any, logrenormf::Real) = renormalize_density(convert(AbstractMeasureOrDensity, density), logrenormf)

renormalize_density(density::AbstractMeasureOrDensity, logrenormf::Real) = Renormalized(density, logrenormf)

renormalize_density(density::Renormalized, logrenormf::Real) = renormalize_density(parent(density), density.logrenormf + logrenormf)
