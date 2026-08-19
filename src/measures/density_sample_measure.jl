# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct DensitySampleMeasure{P,T<:Real,W<:Real,...} <: BATMeasure

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
    dof::Union{IntegerLike,Nothing} = nothing,
    ess::Union{RealLike,Nothing} = nothing,
    mass::Union{RealLike,MeasureBase.AbstractUnknownMass} = 1,
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
    E<:Union{Real,Nothing},
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
    max_weight = maximum(smpls.weight, init = zero(eltype(smpls.weight)))
    DensitySampleMeasure(
        smpls, max_weight, cumsum(smpls.weight),
        dof, ess, _canonical_mass(mass)
    )
end

# ToDo: logarithmic numbers in Measurements?
_canonical_mass(mass::Measurements.Measurement) = mass
_canonical_mass(mass::Real) = exp(ULogarithmic, _lfloat(log(mass)))
_canonical_mass(mass::MeasureBase.AbstractUnknownMass) = mass

Base.convert(::Type{DensitySampleMeasure}, smpls::DensitySampleVector) = DensitySampleMeasure(smpls)
Base.convert(::Type{BATMeasure}, smpls::DensitySampleVector) = DensitySampleMeasure(smpls)

DensitySampleVector(m::DensitySampleMeasure) = deepcopy(samplesof(m))
Base.convert(::Type{DensitySampleVector}, m::DensitySampleMeasure) = DensitySampleVector(m)


function Base.:(==)(a::DensitySampleMeasure, b::DensitySampleMeasure)
    return a._smpls == b._smpls && a._dof == b._dof && a._mass == b._mass
end

function Base.isapprox(a::DensitySampleMeasure, b::DensitySampleMeasure; kwargs...)
    return isapprox(a._smpls, b._smpls; kwargs...) && a._dof == b._dof && a._mass == b._mass
end

# ToDo: Support efficient logdensity lookup. 
function DensityInterface.logdensityof(::DensitySampleMeasure, ::Any)
    throw(ArgumentError("logdensityof is not supported for DensitySampleMeasure."))
end

MeasureBase.getdof(dsm::DensitySampleMeasure) = choose_something(dsm._dof, MeasureBase.NoDOF{typeof(dsm)}())
MeasureBase.massof(dsm::DensitySampleMeasure) = choose_something(dsm._mass, MeasureBase.UnknownMass())

getess(dsm::DensitySampleMeasure) = dsm._ess

empiricalof(dsm::DensitySampleMeasure) = dsm


ValueShapes.varshape(dsm::DensitySampleMeasure) = varshape(samplesof(dsm))

function ValueShapes.unshaped(dsm::DensitySampleMeasure, vs::AbstractValueShape)
    smpls = samplesof(dsm)
    varshape(smpls) <= vs || throw(ArgumentError("Sample shape $(varshape(smpls)) is not compatible with given shape $vs"))
    new_smpls = unshaped.(smpls)
    return DensitySampleMeasure(new_smpls, dsm._max_weight, dsm._cumulative_weight, dsm._dof, dsm._ess, dsm._mass)
end

@inline samplesof(dsm::DensitySampleMeasure) = dsm._smpls


# Reweighting shifts the recorded density values of the samples along with
# the mass, so that the sample logd stays consistent with the density of
# the reweighted measure (sample logd is allowed to be NaN, e.g. after
# transformations with unknown LADJ):
MeasureBase.weightedmeasure(logweight::Real, dsm::DensitySampleMeasure) = _renormalize_empirical_logd(logweight, dsm)


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
    isempty(CW) && throw(ArgumentError("Can't draw from an empty DensitySampleMeasure"))
    W_total = CW[end]
    isfinite(W_total) && W_total > 0 || throw(ArgumentError("Sample weights must sum to a finite positive value"))
    r = rand(get_rng(gen)) * W_total
    idx = searchsortedfirst(CW, r)
    return idx
end

function _rand_subsample_idxs(gen::GenContext, dsm::DensitySampleMeasure, n::Integer)
    # TODO: Use PSIS.

    iszero(n) && return Int[]
    CW = dsm._cumulative_weight
    isempty(CW) && throw(ArgumentError("Can't draw from an empty DensitySampleMeasure"))
    W_total = CW[end]
    isfinite(W_total) && W_total > 0 || throw(ArgumentError("Sample weights must sum to a finite positive value"))
    # Always generate R on CPU for now:
    R = rand(get_rng(gen), n) .* W_total
    idxs = searchsortedfirst.(Ref(CW), R)
    return idxs
end

@inline supports_rand(::DensitySampleMeasure) = true


function MeasureBase.testvalue(::Type{T}, m::DensitySampleMeasure) where {T}
    isempty(m._smpls) && throw(ArgumentError("An empty DensitySampleMeasure has no test value"))
    convert_numtype(T, first(m._smpls.v))
end

function MeasureBase.testvalue(m::DensitySampleMeasure)
    isempty(m._smpls) && throw(ArgumentError("An empty DensitySampleMeasure has no test value"))
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


_renormalize_empirical_logd(::Real, ::Nothing) = nothing

function _renormalize_empirical_logd(logrenorm::Real, dsm::DensitySampleMeasure)
    smpls = samplesof(dsm)
    new_mass = _reweighted_mass(logrenorm, dsm._mass)
    new_smpls = DensitySampleVector((smpls.v, smpls.logd .+ logrenorm, smpls.weight, smpls.info, smpls.aux))
    return DensitySampleMeasure(new_smpls, dsm._max_weight, dsm._cumulative_weight, dsm._dof, dsm._ess, new_mass)
end


# Index-based resampling applies the same indices to both representations
# of a BispacedMeasure empirical, so the pair stays coherent without any
# transform work:
function _unweighted_resampling_byidxs(p::BispacedMeasure, resampled_idxs::AbstractVector{<:Integer})
    BispacedMeasure(
        _unweighted_resampling_byidxs(p.main, resampled_idxs),
        isnothing(p.transformed) ? nothing : _unweighted_resampling_byidxs(p.transformed, resampled_idxs),
        p.f_hash
    )
end

function _unweighted_resampling_byidxs(dsm::DensitySampleMeasure, resampled_idxs::AbstractVector{<:Integer})
    smpls = samplesof(dsm)
    picked = smpls[resampled_idxs]
    # Rebuild instead of overwriting the weights, the weight vector may be
    # immutable (e.g. a Fill):
    new_samples = DensitySampleVector((
        picked.v, picked.logd,
        ones(eltype(picked.weight), length(picked)),
        picked.info, picked.aux,
    ))
    old_ess = getess(dsm)
    # Resampling n_new draws with replacement from a population carrying
    # old_ess effective draws yields about old_ess * n_new / (n_new + old_ess)
    # effective draws (bounded by both old_ess and n_new); the stored entry
    # counts themselves say nothing about effective information:
    n_new = length(new_samples)
    new_ess = isnothing(old_ess) ? nothing : old_ess * n_new / (n_new + old_ess)
    return DensitySampleMeasure(new_samples, dof = dsm._dof, ess = new_ess, mass = massof(dsm))
end
