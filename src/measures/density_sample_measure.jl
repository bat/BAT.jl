# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    DensitySampleMeasure

*BAT-internal, not part of stable public API.*
"""
struct DensitySampleMeasure{SV<:DensitySampleVector,CW} <: BATMeasure
    _smpls::SV
    _cw::CW
end

function DensitySampleMeasure(smpls::DensitySampleVector)
    cw = cumsum(smpls.weight)
    cw .*= inv(cw[end])
    return DensitySampleMeasure(smpls, cw)
end


BATMeasure(smpls::DensitySampleVector) = DensitySampleMeasure(smpls)

DensitySampleVector(m::DensitySampleMeasure) = m._smpls
Base.convert(::Type{DensitySampleVector}, m::DensitySampleMeasure) = DensitySampleVector(m)


function Base.rand(gen::GenContext, ::Type{T}, m::DensitySampleMeasure) where {T}
    r = rand(get_rng(gen))
    idx = searchsortedfirst(m._cw, r)
    return gen_adapt(gen, m._smpls.v[idx])
end

function MeasureBase.testvalue(::Type{T}, m::DensitySampleMeasure) where {T}
    convert_numtype(T, first(m._smpls.v))
end

function MeasureBase.testvalue(m::DensitySampleMeasure)
    first(m._smpls.v)
end

@inline supports_rand(::DensitySampleMeasure) = true


DensityInterface.logdensityof(m::DensitySampleMeasure, x) = NaN
