# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BATSuperpositionMeasure <: BATMeasure

*BAT-internal, not part of stable public API.*

Superposition (sum) of measures.

# Extended help

The log-density of the superposition is computed directly via `logaddexp` over
the component log-densities in an autodiff-friendly way.

Superposing equal BAT measures does not collapse them, for type stability.

`MeasureBase.superpose` generates a BATSuperpositionMeasure if its first
argument is a `BATMeasure`, all other arguments must then be a `BATMeasure`
as well.
"""
struct BATSuperpositionMeasure{C<:Tuple{Vararg{BATMeasure}}} <: BATMeasure
    components::C
end

BATMeasure(m::SuperpositionMeasure{<:Union{Tuple,NamedTuple}}) =
    BATSuperpositionMeasure((map(batmeasure, values(m.components))...,))

Base.:(==)(a::BATSuperpositionMeasure, b::BATSuperpositionMeasure) = a.components == b.components

function Base.show(io::IO, m::BATSuperpositionMeasure)
    print(io, Base.typename(typeof(m)).name, "(")
    show(io, m.components)
    print(io, ")")
end


_superpos_components(m::BATMeasure) = (m,)
_superpos_components(m::BATSuperpositionMeasure) = m.components

function _bat_superpose(a::BATMeasure, b::BATMeasure)
    BATSuperpositionMeasure((_superpos_components(a)..., _superpos_components(b)...))
end


MeasureBase.superpose(a::BATMeasure) = a

function MeasureBase.superpose(a::BATMeasure, bs::AbstractMeasure...)
    all(b -> b isa BATMeasure, bs) || throw(ArgumentError("Can't superpose BAT measures with other measure types, convert them via batmeasure first"))
    return foldl(_bat_superpose, bs, init = a)
end

# Disambiguation with the equal-type method of MeasureBase:
MeasureBase.superpose(a::T, b::T) where {T<:BATMeasure} = _bat_superpose(a, b)


MeasureBase.massof(m::BATSuperpositionMeasure) = sum(massof, m.components)

MeasureBase.testvalue(::Type{T}, m::BATSuperpositionMeasure) where T = testvalue(T, first(m.components))


function DensityInterface.logdensityof(m::BATSuperpositionMeasure, v::Any)
    reduce(_logaddexp, map(Base.Fix2(logdensityof, v), m.components))
end

function checked_logdensityof(m::BATSuperpositionMeasure, v::Any)
    reduce(_logaddexp, map(Base.Fix2(checked_logdensityof, v), m.components))
end


supports_rand(m::BATSuperpositionMeasure) = all(supports_rand, m.components)

function Base.rand(gen::GenContext, m::BATSuperpositionMeasure)
    supports_rand(m) || throw(ArgumentError("Superposition components must support rand"))
    w = map(c -> float(massof(c)), m.components)
    W = sum(w)
    # Guards the component walk below, an infinite or zero total mass would
    # silently sample the last component. We'll assume the components themselves
    # are finite, as testing for finite mass is expensive and as rand will
    # automatically fail otherwise anyhow.
    isfinite(W) && W > 0 || throw(ArgumentError("Superposition components must have finite total mass to support rand"))
    u = rand(get_rng(gen)) * W
    return _rand_superpos_component(gen, m.components, w, u)
end

_rand_superpos_component(gen::GenContext, cs::Tuple{BATMeasure}, w::Tuple, u::Real) = rand(gen, only(cs))

function _rand_superpos_component(gen::GenContext, cs::Tuple, w::Tuple, u::Real)
    # Strict comparison, so zero-mass components are skipped even at u == 0:
    u < first(w) ? rand(gen, first(cs)) : _rand_superpos_component(gen, Base.tail(cs), Base.tail(w), u - first(w))
end


function MeasureBase.getdof(m::BATSuperpositionMeasure)
    dof = getdof(first(m.components))
    @argcheck all(c -> getdof(c) == dof, m.components)
    return dof
end


function ValueShapes.varshape(m::BATSuperpositionMeasure)
    vs = varshape(first(m.components))
    @argcheck all(c -> varshape(c) == vs, m.components)
    return vs
end

ValueShapes.unshaped(m::BATSuperpositionMeasure) = BATSuperpositionMeasure(map(unshaped, m.components))

(shape::AbstractValueShape)(m::BATSuperpositionMeasure) = BATSuperpositionMeasure(map(shape, m.components))


has_uhc_support(m::BATSuperpositionMeasure) = all(has_uhc_support, m.components)
