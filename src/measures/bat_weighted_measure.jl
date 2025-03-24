# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BATWeightedMeasure <: BATMeasure

*BAT-internal, not part of stable public API.*
"""
struct BATWeightedMeasure{T<:Real,D<:BATMeasure} <: BATMeasure
    logweight::T
    base::D
end

BATMeasure(m::WeightedMeasure) = BATWeightedMeasure(m.logweight, batmeasure(m.base))

Base.:(==)(a::BATWeightedMeasure, b::BATWeightedMeasure) = a.base == b.base && a.logweight == b.logweight


MeasureBase.weightedmeasure(logweight::Real, m::BATMeasure) = _bat_weightedmeasure(logweight, m)

_bat_weightedmeasure(logweight::Real, m::BATMeasure) = BATWeightedMeasure(logweight, m)

_bat_weightedmeasure(logweight::Real, m::BATWeightedMeasure) = weightedmeasure(m.logweight + logweight, m.base)


MeasureBase.basemeasure(m::BATWeightedMeasure) = m.base


function Base.show(io::IO, d::BATWeightedMeasure)
    print(io, Base.typename(typeof(d)).name, "(")
    show(io, d.logweight)
    print(io, ", ")
    show(io, d.base)
    print(io, ")")
end


function DensityInterface.logdensityof(m::BATWeightedMeasure, v::Any)
    parent_logd = logdensityof(m.base,v)
    R = float(typeof(parent_logd))
    convert(R, parent_logd + m.logweight)
end

function checked_logdensityof(m::BATWeightedMeasure, v::Any)
    parent_logd = checked_logdensityof(m.base,v)
    R = float(typeof(parent_logd))
    convert(R, parent_logd + m.logweight)
end


Base.rand(gen::GenContext, m::BATWeightedMeasure) = rand(gen, m.base)

supports_rand(m::BATWeightedMeasure) = supports_rand(m.origin)


Statistics.mean(m::BATWeightedMeasure) = mean(m.base)
Statistics.var(m::BATWeightedMeasure) = var(m.base)
Statistics.cov(m::BATWeightedMeasure) = cov(m.base)


measure_support(m::BATWeightedMeasure) = measure_support(m.base)


ValueShapes.varshape(m::BATWeightedMeasure) = varshape(m.base)

ValueShapes.unshaped(m::BATWeightedMeasure) = weightedmeasure(m.logweight, unshaped(m.base))

(shape::AbstractValueShape)(m::BATWeightedMeasure) = weightedmeasure(m.logweight, shape(m.base))



# ToDo: This should just be a method of a proper `bat_renormalize`` API function
# when using an `AutoRenormalize` (or similar name) algorithm:
"""
    BAT.auto_renormalize(measure::MeasureBase.AbstractMeasure)

*Experimental feature, not part of stable public API.*

Returns `(result = new_measure, logweight = logweight)`.

Tries to automatically renormalize `measure` if a maxium log-m value
is available, returns `measure` unchanged otherwise.
"""
function auto_renormalize(measure::AbstractMeasure)
    _generic_auto_renormalize_impl(_estimated_max_logd(measure), batmeasure(measure))
end


_estimated_max_logd(::AbstractMeasure) = missing
_estimated_max_logd(::Nothing) = missing

function _estimated_max_logd(samples::DensitySampleVector)
    logweight = maximum(samples.logd)
    isnan(logweight) || isinf(logweight) ? zero(logweight) : logweight
end

function _generic_auto_renormalize_impl(max_logd::Real, measure::AbstractMeasure)
    logweight = - max_logd
    result = weightedmeasure(logweight, measure)
    (result = result, logweight = logweight)
end

function _generic_auto_renormalize_impl(::Missing, measure::AbstractMeasure)
    (result = measure, logweight = false)
end


_dist_with_pushfwd(m::BATWeightedMeasure) = _dist_with_pushfwd_impl(m.base, identity)
_dist_with_pullback(m::BATWeightedMeasure) = _dist_with_pullback_impl(m.base, identity)
