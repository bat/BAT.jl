# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct DensitySampleMeasureP,T<:Real,W<:Real,...} <: BATMeasure

Represents an
[Empirical Measure](https://en.wikipedia.org/wiki/Empirical_measure)
based on a sample of points (of type `P` with weights of type `W`) drawn from
a normalizable measure, with the log-density values (of type `T`) of that
measure at the sample points stored as well.

The sample need not have been drawn in a true IID fashion, but may also be
the result of MCMC and other sampling methods.

A `DensitySampleMeasure` can be converted to a `DensitySampleVector`.

Note: `DensitySampleMeasure` currently does not support `logdensityof`, as it
would require an inefficient linear search over all sample points.

Constructors:

```julia
function DensitySampleMeasure(
    smpls::DensitySampleVector;
    dof::Union{Number,Nothing},
    mass::Union{Number,MeasureBase.AbstractUnknownMass},
)
```

A `DensitySampleMeasure` has mass one by default, as the measure the samples
were drawn from is treated as implicitly normalized, even if it was a
scaled probability measure of possibly unknown total mass (e.g. a
non-normalized Bayesian posterior measure).
"""
struct DensitySampleMeasure{
    P,
    T<:Real,
    W<:Real,
    SV<:DensitySampleVector{P,T,W},
    WV<:AbstractVector{W},
    N<:Union{IntegerLike,Nothing},
    N<:Union{Real,Nothing},
    U<:Union{Real,MeasureBase.AbstractUnknownMass}
} <: BATMeasure
    _smpls::SV
    _max_weight::W
    _cumulative_weight::WV
    _dof::N
    _ess::E
    _mass::U
end
export DensitySampleMeasure


function DensitySampleMeasure(
    smpls::DensitySampleVector;
    dof::Union{IntegerLike,Nothing} = nothing,
    ess::Union{RealLike,Nothing} = nothing,
    mass::Union{RealLike,MeasureBase.AbstractUnknownMass} = 1,
)
    # ToDo: Ensure smpls are deduplicated.
    # ToDo: Enable logdensity calculation by storing a binary searchable vector
    # over tuples `(point_hash, sample_idx)`?
    DensitySampleMeasure(
        smpls, maximum(smpls.weight), cumsum(smpls.weight),
        dof, ess, _canonical_mass(mass)
    )
end

# ToDo: logarithmic numbers in Measurements?
_canonical_mass(mass::Measurements.Measurement) = mass
_canonical_mass(mass::Real) = exp(ULogarithmic, _lfloat(log(mass)))

Base.convert(::Type{DensitySampleMeasure}, smpls::DensitySampleVector) = DensitySampleMeasure(smpls)

DensitySampleVector(m::DensitySampleMeasure) = deepcopy(samplesof(m))
Base.convert(::Type{DensitySampleVector}, m::DensitySampleMeasure) = DensitySampleVector(m)


function Base.:(==)(a::DensitySampleMeasure, b::DensitySampleMeasure)
    return a._smpls == b._smpls && a._dof == b._dof && a._mass == b._mass
end

function Base.isapprox(a::DensitySampleMeasure, b::DensitySampleMeasure; kwargs...)
    return isapprox(a._smpls, b._smpls; kwargs...) && isapprox(a._dof, b._dof; kwargs...) &&
        isapprox(a._mass, b._mass; kwargs...)
end

# ToDo: Support efficient logdensity lookup. 
function DensityInterface.logdensityof(::DensitySampleMeasure, ::Any)
    throw(ArgumentError("logdensityof is not supported for DensitySampleMeasure."))
end

MeasureBase.getdof(dsm::DensitySampleMeasure) = choose_something(dsm._dof, MeasureBase.NoDOF{typeof(dsm)}())
MeasureBase.massof(dsm::DensitySampleMeasure) = choose_something(dsm._mass, MeasureBase.UnknownMass())


ValueShapes.varshape(dsm::DensitySampleMeasure) = varshape(samplesof(dsm))

function ValueShapes.unshaped(dsm::DensitySampleMeasure, vs::AbstractValueShape)
    smpls = samplesof(dsm)
    @assert varshape(smpls) <= vs
    new_smpls = unshaped.(smpls)
    return DensitySampleMeasure(new_smpls, dsm.approx, dsm.mass, dsm.modes, dsm.samplegen)
end

# ToDo: Return some kind of implicit support object?
# measure_support(dsm::DensitySampleMeasure) = ...

@inline samplesof(dsm::DensitySampleMeasure) = dsm._smpls


function MeasureBase.weightedmeasure(logweight::Real, dsm::DensitySampleMeasure)
    new_mass = _reweighted_mass(logweight, dsm._mass)
    return DensitySampleMeasure(dsm._smpls, dsm._weight_sum, dsm._max_weight, dsm._dof, new_mass)
end


function Base.show(io::IO, ::MIME"text/plain", dsm::DensitySampleMeasure)
    if get(io, :compact, false)
        print(io, "DensitySampleMeasure(...)")
    else
        println(io, "DensitySampleMeasure:")
        show(io, samplesof(dsm))
    end
 end


function Base.rand(gen::GenContext, dsm::DensitySampleMeasure)
    idx = _rand_subsample_idx(gen, dsm)
    return gen_adapt(gen, dsm._smpls.v[idx])
end

function _rand_subsample_idx(gen::GenContext, dsm::DensitySampleMeasure)
    # TODO: Use PSIS.

    CW = dsm._cumulative_weight
    r = rand(get_rng(gen)) * CW[end]
    idx = searchsortedfirst(dsm._cumulative_weight, r)
    return idx
end

function _rand_subsample_idxs(gen::GenContext, dsm::DensitySampleMeasure, n::Integer)
    # TODO: Use PSIS.

    CW = dsm._cumulative_weight
    # Always generate R on CPU for now:
    R = rand(get_rng(gen), n) .* CW[end]
    idxs = searchsortedfirst.(Ref(CW), R)
    return idxs
end

@inline supports_rand(::DensitySampleMeasure) = true


function MeasureBase.testvalue(::Type{T}, m::DensitySampleMeasure) where {T}
    convert_numtype(T, first(m._smpls.v))
end

function MeasureBase.testvalue(m::DensitySampleMeasure)
    first(m._smpls.v)
end


function LazyReports.pushcontent!(rpt::LazyReport, dsm::DensitySampleMeasure)
    lazyreport!(rpt, samplesof(dsm))
end



Statistics.mean(dsm::DensitySampleMeasure) = mean(samplesof(dsm))
Statistics.median(dsm::DensitySampleMeasure) = median(samplesof(dsm))
Statistics.var(dsm::DensitySampleMeasure) = var(samplesof(dsm))
Statistics.std(dsm::DensitySampleMeasure) = std(samplesof(dsm))
Statistics.cov(dsm::DensitySampleMeasure) = cov(samplesof(dsm))

_approx_mean(dsm::DensitySampleMeasure, n) = mean(dsm)
_approx_cov(dsm::DensitySampleMeasure, n) = cov(dsm)


function _approx_max_logd(dsm::DensitySampleMeasure)
    smpls = samplesof(dsm)
    @assert !isnothing(smpls)
    return _approx_max_logd(smpls)
end


_renormalize_empirical_logd(::Missing) = missing

function _renormalize_empirical_logd(logrenorm::Real, dsm::DensitySampleMeasure)
    smpls = samplesof(dsm)
    new_mass = _reweighted_mass(logrenorm, dsm._mass)
    new_smpls = DensitySampleVector((smpls.v, smpls.logd .+ logrenorm, smpls.weight, smpls.info, smpls.aux))
    return DensitySampleMeasure(new_smpls, dsm._max_weight, dsm._cumulative_weight, dsm._dof, new_mass)
end


function _unweighted_resampling_byidxs(dsm::DensitySampleMeasure, resampled_idxs::AbstractVector{<:Integer})
    smpls = samplesof(dsm)
    new_samples = smpls[resampled_idxs]
    new_samples.weight .= 1
    old_ess = getess(dsm)
    new_ess = isnothing(old_ess) ? nothing : old_ess * (length(new_samples) / length(smpls))
    new_dsm = DensitySampleMeasure(new_smpls, dof = getdof(dsm), ess = new_ess, mass = massof(dsm))
    return new_dsm
end
