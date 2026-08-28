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

The measure keeps a reference to `smpls` (accessible via `samplesof`) and
always reflects its live weight values: random generation and statistics
are consistent with the current state of the sample vector.

Note: `DensitySampleMeasure` does not support `logdensityof`. An empirical
measure has no density in the usual sense, the log-density values of the
original measure at the sample points are available via
`samplesof(dsm).logd`.

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
    N<:Union{IntegerLike,Nothing},
    E<:Union{Real,Nothing},
    U<:Union{Real,MeasureBase.AbstractUnknownMass}
} <: BATMeasure
    _smpls::SV
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
    # Empirical-measure weights must support categorical sampling: a
    # negative or non-finite weight would make the subsampling CDF
    # non-monotone, an all-zero weight vector would leave nothing to draw
    # (the sample vector is shared, not copied, so draw-time code
    # revalidates against later weight mutation - but constructing an
    # invalid empirical measure should fail loudly right away):
    W = smpls.weight
    all(w -> isfinite(w) && w >= 0, W) || throw(ArgumentError(
        "Weights of an empirical measure must be finite and non-negative"
    ))
    isempty(W) || maximum(W) > 0 || throw(ArgumentError(
        "Weights of an empirical measure must contain at least one strictly positive entry"
    ))
    DensitySampleMeasure(smpls, dof, ess, _canonical_mass(mass))
end

# Masses are stored on a canonical logarithmic Float64 scale, uniformly
# across all integration and sampling backends: the log scale keeps very
# small and very large masses representable without extended-range number
# types. Uncertainties are transported to the log scale to first order:
_canonical_mass(mass::ULogarithmic) = mass
_canonical_mass(mass::Real) = exp(ULogarithmic, Float64(log(mass)))
function _canonical_mass(mass::Measurements.Measurement)
    val = Measurements.value(mass)
    unc = Measurements.uncertainty(mass)
    return exp(ULogarithmic, Measurements.measurement(Float64(log(val)), Float64(unc / val)))
end
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

# An empirical measure has no density with respect to Lebesgue. What a
# point lookup could return is the recorded log-density of the measure the
# samples were drawn from, which is directly available as
# `samplesof(dsm).logd`:
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
    return DensitySampleMeasure(new_smpls, dsm._dof, dsm._ess, dsm._mass)
end

# Disambiguates against unshaped(x, ::ConstValueShape) of ValueShapes:
ValueShapes.unshaped(dsm::DensitySampleMeasure, vs::ConstValueShape) =
    invoke(unshaped, Tuple{DensitySampleMeasure,AbstractValueShape}, dsm, vs)

@inline samplesof(dsm::DensitySampleMeasure) = dsm._smpls


# Empirical representations of one pushforward share probability masses.
function _with_sample_weights(dsm::DensitySampleMeasure, weights::AbstractVector{<:Real})
    smpls = samplesof(dsm)
    smpls.weight === weights && return dsm
    new_smpls = DensitySampleVector((smpls.v, smpls.logd, weights, smpls.info, smpls.aux))
    return DensitySampleMeasure(new_smpls, dsm._dof, dsm._ess, dsm._mass)
end

_empirical_weights_shared(p::BispacedMeasure{<:DensitySampleMeasure,<:DensitySampleMeasure}) =
    samplesof(p.main).weight === samplesof(p.transformed).weight


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

# The subsampling CDF is computed fresh from the live sample weights on
# each draw call: the sample vector is shared with the caller (it is
# user-facing via `samplesof`), so a cached CDF could silently
# desynchronize from mutated weights. Canonical relative weights make
# the CDF monotone, finite and rescaling-invariant, and revalidate the
# weights against invalid mutation:
function _live_weight_cdf(dsm::DensitySampleMeasure)
    W = samplesof(dsm).weight
    isempty(W) && throw(ArgumentError("Can't draw from an empty DensitySampleMeasure"))
    rel_weights = _canonical_rel_weights(W)
    cdf = similar(rel_weights, _weight_accum_type(rel_weights))
    copyto!(cdf, rel_weights)
    return cumsum!(cdf, cdf)
end

function _rand_subsample_idx(gen::GenContext, dsm::DensitySampleMeasure)
    # TODO: Use PSIS.

    CW = _live_weight_cdf(dsm)
    r = rand(get_rng(gen)) * CW[end]
    idx = searchsortedfirst(CW, r)
    return idx
end

function _rand_subsample_idxs(gen::GenContext, dsm::DensitySampleMeasure, n::Integer)
    # TODO: Use PSIS.

    iszero(n) && return Int[]
    CW = _live_weight_cdf(dsm)
    # Always generate R on CPU for now:
    R = rand(get_rng(gen), n) .* CW[end]
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
    return DensitySampleMeasure(new_smpls, dsm._dof, dsm._ess, new_mass)
end


# Multinomial resampling destroys the process order, so per-sample
# process provenance (MCMC sample ids) must not survive it - unlike
# order-preserving systematic resampling, which keeps its ids:
function _without_sampleids(dsm::DensitySampleMeasure)
    s = samplesof(dsm)
    new_s = DensitySampleVector((s.v, s.logd, s.weight, fill(nothing, length(eachindex(s))), s.aux))
    return DensitySampleMeasure(new_s, dsm._dof, dsm._ess, dsm._mass)
end

_without_sampleids(::Nothing) = nothing

_without_sampleids(p::BispacedMeasure) =
    BispacedMeasure(_without_sampleids(p.main), _without_sampleids(p.transformed), p.f_hash)

# Index-based resampling applies the same indices to both representations
# of a BispacedMeasure empirical, so the pair stays coherent without any
# transform work:
function _unweighted_resampling_byidxs(
    p::BispacedMeasure,
    resampled_idxs::AbstractVector{<:Integer};
    preserve_ess::Bool = false,
)
    main = _unweighted_resampling_byidxs(p.main, resampled_idxs; preserve_ess)
    transformed = isnothing(p.transformed) ? nothing :
        _with_sample_weights(
            _unweighted_resampling_byidxs(p.transformed, resampled_idxs; preserve_ess),
            samplesof(main).weight,
        )
    BispacedMeasure(main, transformed, p.f_hash)
end

function _unweighted_resampling_byidxs(
    dsm::DensitySampleMeasure,
    resampled_idxs::AbstractVector{<:Integer};
    preserve_ess::Bool = false,
)
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
    # Random resampling adds conditional Monte Carlo variance, giving the
    # approximate effective count below. Identity systematic resampling
    # adds none:
    n_new = length(new_samples)
    new_ess = preserve_ess || isnothing(old_ess) ? old_ess : old_ess * n_new / (n_new + old_ess)
    return DensitySampleMeasure(new_samples, dof = dsm._dof, ess = new_ess, mass = massof(dsm))
end
