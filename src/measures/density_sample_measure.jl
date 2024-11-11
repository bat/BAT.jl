# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct DensitySampleMeasureP,T<:Real,W<:Real,...} <: BATMeasure

Represents an
[Empirical Measure](https://en.wikipedia.org/wiki/Empirical_measure)
based on a sample of points (of type `P` with weights of type `W`) dawn from
a normalizable measure, with the log-density values (of type `T`) of that
measure at the sample points stored as well.

The sample need not have been drawn in a true IID fashion, but may also be
the result of MCMC and other sampling methods.

A `DensitySampleMeasure` can be converted to a `DensitySampleVector`.

Note: `DensitySampleMeasure` currently does not support `logdensityof`, as it
would require an inefficient linear search over all sample points.

Constructors:

```julia
DensitySampleMeasure(smpls::DensitySampleVector, dof::Integer; mass::Real = 1)
```

A `DensitySampleMeasure` hass mass one by default, as the measure the samples
were drawn from is treated as implicitly normalized, even if is was a
scaled probability measure of possibly unknown total mass (e.g. a
non-normalized Bayesian posterior measure).
"""
struct DensitySampleMeasure{
    P,
    T<:Real,
    W<:Real,
    M<:Real,
    SV<:DensitySampleVector{P,T,W},
    WV<:AbstractVector{W}
} <: BATMeasure
    _smpl::SV
    _max_weight::W
    _cumulative_weight::WV
    _dof::Integer
    _mass::M
end
export DensitySampleMeasure


function DensitySampleMeasure(smpls::DensitySampleVector, dof::Integer; mass::Real = 1)
    # ToDo: Ensure smpls are deduplicated.
    # ToDo: Storing logdensity calcuaion by storing a binary searchable vector
    # over tuples `(point_hash, sample_idx)`.
    conv_mass = exp(ULogarithmic, _lfloat(log(mass)))
    DensitySampleMeasure(smpls, maximum(smpls.weight), cumsum(smpls.weight), dof, conv_mass)
end

Base.convert(::Type{DensitySampleMeasure}, smpls::DensitySampleVector) = DensitySampleMeasure(smpls)

DensitySampleVector(m::DensitySampleMeasure) = samplesof(m)
Base.convert(::Type{DensitySampleVector}, m::DensitySampleMeasure) = DensitySampleVector(m)


function Base.:(==)(a::DensitySampleMeasure, b::DensitySampleMeasure)
    return a._smpl == b._smpl && a._dof == b._dof && a._logmass == b._logmass
end

function Base.isapprox(a::DensitySampleMeasure, b::DensitySampleMeasure; kwargs...)
    return isapprox(a._smpl, b._smpl; kwargs...) && isapprox(a._dof, b._dof; kwargs...) &&
        isapprox(a._logmass, b._logmass; kwargs...)
end

# ToDo: Support efficient logdensity lookup. 
function DensityInterface.logdensityof(::DensitySampleMeasure, ::Any)
    throw(ArgumentError("logdensityof is not supported for DensitySampleMeasure."))
end

MeasureBase.getdof(em::DensitySampleMeasure) = em._dof

MeasureBase.massof(em::DensitySampleMeasure) = em._mass

ValueShapes.varshape(em::DensitySampleMeasure) = elshape(samplesof(em))

function _unshaped_density(em::DensitySampleMeasure, vs::AbstractValueShape)
    new_measure = unshaped(em.measure, vs)
    @assert varshape(em.sampled) <= vs
    new_sampled = unshaped.(em.sampled)
    return DensitySampleMeasure(new_measure, new_sampled, em.approx, em.mass, em.modes, em.samplegen)
end

measure_support(em::DensitySampleMeasure) = measure_support(em.measure)

@inline maybe_empiricalof(em::DensitySampleMeasure) = em.sampled
@inline maybe_samplesof(em::DensitySampleMeasure) = maybe_empiricalof(em) isa Missing ? missing : convert(DensitySamplesVector, em.sampled)
@inline maybe_modesof(em::DensitySampleMeasure) = em.modes


get_initsrc_from_target(em::DensitySampleMeasure) = samplesof(em)


function bat_transform_impl(f_transform, em::DensitySampleMeasure, algorithm::SampleTransformation, context::BATContext)
    smpls = samplesof(em)
    new_samples, _ = bat_transform_impl(f_transform, smpls, algorithm, context)
    return DensitySampleMeasure(new_samples)
end

function MeasureBase.weightedmeasure(logweight::Real, em::DensitySampleMeasure)
    current_logmass = _lfloat(log(em._mass))
    new_lowmass = oftype(current_logmass, logweight) + current_logmass
    new_mass = exp(ULogarithmic, new_lowmass)
    return DensitySampleMeasure(em._smpl, em._weight_sum, em._max_weight, em._dof, new_mass)
end


# function Base.show(io::IO, mime::MIME"text/plain", sd::DensitySampleMeasure)
#     if get(io, :compact, false)
#         println(io, "DensitySampleMeasure(...)")
#     else
#         # ...
#     end
# end


function Base.rand(gen::GenContext, em::DensitySampleMeasure) where {T}
    idx = _rand_subsample_idx(gen, em)
    return gen_adapt(gen, em._smpl.v[idx])
end

function _rand_subsample_idx(gen::GenContext, em::DensitySampleMeasure)
    # TODO: Use PSIS.

    CW = em._cumulative_weight
    r = rand(get_rng(gen)) * CW[end]
    idx = searchsortedfirst(em._cw, r)
    return idx
end

function _rand_subsample_idxs(gen::GenContext, em::DensitySampleMeasure, n::Integer)
    # TODO: Use PSIS.

    CW = em._cumulative_weight
    # Always generate R on CPU for now:
    R = rand(get_rng(gen), n) .* CW[end]
    idxs = searchsortedfirst.(Ref(CW), R)
    return idxs
end

@inline supports_rand(::DensitySampleMeasure) = true


function MeasureBase.testvalue(::Type{T}, m::DensitySampleMeasure) where {T}
    convert_numtype(T, first(m._smpl.v))
end

function MeasureBase.testvalue(m::DensitySampleMeasure)
    first(m._smpl.v)
end



function bat_report!(md::Markdown.MD, em::DensitySampleMeasure)
    if !isnothing(em.sampled)
        bat_report!(md, em.sampled)
    end
    if !isnothing(em.samplegen)
        bat_report!(md, em.samplegen)
    end

    return md
end


 _approx_mean(em::DensitySampleMeasure, n) = mean(samplesof(em))

 _approx_cov(em::DensitySampleMeasure, n) = cov(samplesof(em))

function _estimated_max_logd(em::DensitySampleMeasure)
    smpls = samplesof(em)
    @assert !isnothing(smpls)
    return _estimated_max_logd(smpls)
end


_renormalize_empirical_logd(::Missing) = missing

function _renormalize_empirical_logd(logrenorm::Real, em::DensitySampleMeasure)
    smpls = samplesof(em)
    new_smpls = DensitySampleVector((smpls.v, smpls.logd .+ logrenorm, smpls.weight, smpls.info, smpls.aux))
    return DensitySampleMeasure(new_smpls, em._weight_sum, em._max_weight)
end
