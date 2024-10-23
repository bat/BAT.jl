# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    measure_support(measure::AbstractMeasure)

*BAT-internal, not part of stable public API.*

Get the parameter bounds of `measure`. May return a `IntervalsSets.Domain` or `BAT.UnknownDomain`
"""
function measure_support end

measure_support(m::AbstractMeasure) = UnknownDomain()

struct UnknownDomain end



"""
    abstract type BATMeasure <:AbstractMeasure

*BAT-internal, not part of stable public API.*

Subtypes must implement `DensityInterface.logdensityof` and
`ValueShapes.varshape`.
"""
abstract type BATMeasure <: AbstractMeasure end

Base.convert(::Type{BATMeasure}, m::BATMeasure) = m
Base.convert(::Type{BATMeasure}, m::AbstractMeasure) = BATMeasure(m)
Base.convert(::Type{BATMeasure}, d::Distribution) = BATMeasure(d)

BATMeasure(::StdUniform) = batmeasure(StandardUvUniform())
BATMeasure(::StdNormal) = batmeasure(StandardUvNormal())




function _rv_dof(m::AbstractMeasure)
    tv = testvalue(m)
    if !(tv isa AbstractVector{<:Real})
        throw(ArgumentError("Measure of type $(nameof(typeof(m))) is not on the space of real-valued vectors"))
    end
    length(eachindex(tv))
end


DensityInterface.logdensityof(@nospecialize(m::BATMeasure), ::Any) = throw(ArgumentError("logdensityof not implemented for $(typeof(m))"))

MeasureBase.logdensity_def(m::BATMeasure, ::Any) = throw(ArgumentError("logdensity_def not implemented for $(typeof(m))"))
MeasureBase.basemeasure(m::BATMeasure) = throw(ArgumentError("basemeasure not implemented for $(typeof(m))"))
MeasureBase.rootmeasure(m::BATMeasure) = throw(ArgumentError("rootmeasure not implemented for $(typeof(m))"))
MeasureBase.massof(::BATMeasure) = MeasureBase.UnknownMass()

# ToDo: Use `x in measure_support(m)` later on, when MeasureBase calls `insupport` less often:
@static if isdefined(MeasureBase, :NoFastInsupport)
    MeasureBase.insupport(m::BATMeasure, ::Any) = MeasureBase.NoFastInsupport{typeof(m)}()
else
    # Workaround:
    MeasureBase.insupport(m::BATMeasure, ::Any) = true
end

@static if isdefined(MeasureBase, :localmeasure)
    MeasureBase.localmeasure(m::BATMeasure, ::Any) = m
end


# ToDo: Specialize for (e.g.) DensitySampleMeasure:
_default_measure_precision(::BATMeasure) = Float64

# ToDo: Specialize for certain measures?
_default_cunit(::BATMeasure) = CPUnit()

function Base.rand(rng::AbstractRNG, ::Type{T}, m::BATMeasure) where {T<:Real}
    cunit = _default_cunit(m)
    rand(GenContext{T}(cunit, rng), m)
end

function Base.rand(rng::AbstractRNG, m::BATMeasure)
    rand(rng, _default_measure_precision(m), m)
end


function ValueShapes.unshaped(measure::BATMeasure, vs::AbstractValueShape)
    varshape(measure) <= vs || throw(ArgumentError("Shape of measure not compatible with given shape"))
    unshaped(measure)
end


show_value_shape(io::IO, vs::AbstractValueShape) = show(io, vs)
function show_value_shape(io::IO, vs::NamedTupleShape)
    print(io, Base.typename(typeof(vs)).name, "(")
    show(io, propertynames(vs))
    print(io, "}(…)")
end

function Base.show(io::IO, d::BATMeasure)
    print(io, Base.typename(typeof(d)).name, "(objectid = ")
    show(io, objectid(d))
    vs = varshape(d)
    if !ismissing(vs)
        print(io, ", varshape = ")
        show_value_shape(io, vs)
    end
    print(io, ")")
end


"""
    batmeasure(obj)

*Experimental feature, not part of stable public API.*

Convert a measure-like `obj` to a measure that is compatible with BAT.
"""
function batmeasure end
export batmeasure

batmeasure(obj) = convert(BATMeasure, obj)


"""
    batsampleable(obj)

*Experimental feature, not part of stable public API.*

Convert `obj` into something that BAT can sample from.
"""
batsampleable(obj) = batmeasure(obj)
batsampleable(smpls::DensitySampleVector) = smpls


function convert_for(operation::Function, target)
    try
        batmeasure(target)
    catch err
        throw(ArgumentError("Can't convert $operation target of type $(nameof(typeof(target))) to a BAT-compatible measure."))
    end
end


"""
    supports_rand(m)

*BAT-internal, not part of stable public API.*

Convert a measure-like object `m` supports `rand`.
"""
@inline supports_rand(::AbstractMeasure) = false
@inline supports_rand(::StdMeasure) = true
@inline supports_rand(m::WeightedMeasure) = supports_rand(m.base)
@inline supports_rand(m::PushforwardMeasure) = !(gettransform(m) isa NoInverse) && supports_rand(transport_origin(m))



"""
    measure_support(
        measure::BATMeasure
    )::Union{AbstractVarBounds,Missing}

*BAT-internal, not part of stable public API.*

Get the parameter bounds of `measure`. See [`BATMeasure`](@ref) for the
implications and handling of bounds.
"""
measure_support(::BATMeasure) = missing

_is_uhc(::UnitCube) = true
_is_uhc(d::Rectangle) = isapproxzero(d.a) && isapproxone(d.b)

has_uhc_support(m::BATMeasure) = _is_uhc(measure_support(m))

is_std_mvnormal(::AbstractMeasure) = false
is_std_mvnormal(::MeasureBase.PowerMeasure{MeasureBase.StdNormal}) = true

ValueShapes.varshape(::BATMeasure) = missing


MeasureBase.transport_to(mu::Union{Distribution,AbstractMeasure}, nu::BATMeasure) = _bat_transport_to(batmeasure(mu), nu)
MeasureBase.transport_to(mu::BATMeasure, nu::Union{Distribution,AbstractMeasure}) = _bat_transport_to(mu, batmeasure(nu))
MeasureBase.transport_to(mu::BATMeasure, nu::BATMeasure) = _bat_transport_to(mu, nu)

function _bat_transport_to(mu, nu)
    target_dist, target_pushfwd = _dist_with_pushfwd(mu)
    source_dist, source_pullback = _dist_with_pullback(nu)
    trafo = DistributionTransform(target_dist, source_dist)
    return fcomp(target_pushfwd, fcomp(trafo, source_pullback))
end

_dist_with_pushfwd(m::BATMeasure) = Distribution(m), identity

function _dist_with_pushfwd_impl(origin, f)
    d, g = _dist_with_pushfwd(origin)
    d, fcomp(f, g)
end

function _combine_dwp_with_f(dwp, f)
    d, g = dwp
    return d, fcomp(f, g)
end

_dist_with_pullback(m::BATMeasure) = Distribution(m), identity

function _dist_with_pullback_impl(origin, finv)
    d, ginv = _dist_with_pullback(origin)
    return d, fcomp(ginv, finv)
end



"""
    BAT.MeasureLike = Union{...}

*BAT-internal, not part of stable public API.*

Union of all types that BAT will accept as a measures or convert to measures.
"""
const MeasureLike = Union{
    MeasureBase.AbstractMeasure,
    Distributions.Distribution,
    BAT.DensitySampleVector
}

# !!!!!! Remove AnySampleable and provide conversion from samples to measure

"""
    BAT.AnySampleable = Union{...}

Union of all types that BAT can sample from:

* [`BAT.MeasureLike`](@ref)
* [`BAT.DensitySampleVector`](@ref)
"""
const AnySampleable = Union{
    BAT.MeasureLike,
}
export AnySampleable
