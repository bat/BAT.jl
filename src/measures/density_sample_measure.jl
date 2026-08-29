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

The measure snapshots the sampling weights at construction and builds a
private cumulative distribution. Its values, log densities, `info`, and `aux`
columns remain shared with `smpls`, but its sampling weights do not. Later
weight changes require constructing a replacement `DensitySampleMeasure`.
In particular, the live data returned by `samplesof` must not be modified:
mutating its owned weights would desynchronize its cached sampling CDF.

The stored effective sample size (`ess`) records sampling-process
provenance, not empirical-measure content. It is available through
[`getess`](@ref), but does not participate in equality or hashing.

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
    CW<:AbstractVector{<:Real},
    N<:Union{IntegerLike,Nothing},
    E<:Union{Real,Nothing},
    U<:Union{Real,MeasureBase.AbstractUnknownMass}
} <: BATMeasure
    _smpls::SV
    _cumulative_weight::CW
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
    # Empirical-measure weights must support categorical sampling: a negative
    # or non-finite weight would make the subsampling CDF non-monotone, and
    # an all-zero vector would leave nothing to draw. Copy only this column:
    # the owned weights and CDF make repeated scalar draws independent of the
    # caller's mutable weight storage while retaining the sample payload.
    W = smpls.weight
    all(w -> isfinite(w) && w >= 0, W) || throw(ArgumentError(
        "Weights of an empirical measure must be finite and non-negative"
    ))
    isempty(W) || maximum(W) > 0 || throw(ArgumentError(
        "Weights of an empirical measure must contain at least one strictly positive entry"
    ))
    owned_weights = copy(W)
    owned_smpls = DensitySampleVector((smpls.v, smpls.logd, owned_weights, smpls.info, smpls.aux))
    DensitySampleMeasure(owned_smpls, _weight_cdf(owned_weights), dof, ess, _canonical_mass(mass))
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

function Base.isequal(a::DensitySampleMeasure, b::DensitySampleMeasure)
    a_smpls, b_smpls = a._smpls, b._smpls
    return isequal(a_smpls.v, b_smpls.v) && isequal(a_smpls.logd, b_smpls.logd) &&
        isequal(a_smpls.weight, b_smpls.weight) && isequal(a_smpls.info, b_smpls.info) &&
        isequal(a_smpls.aux, b_smpls.aux) && isequal(a._dof, b._dof) && isequal(a._mass, b._mass)
end

function Base.hash(dsm::DensitySampleMeasure, h::UInt)
    smpls = dsm._smpls
    h = hash(:DensitySampleMeasure, h)
    h = hash(smpls.v, h)
    h = hash(smpls.logd, h)
    h = hash(smpls.weight, h)
    h = hash(smpls.info, h)
    h = hash(smpls.aux, h)
    h = hash(dsm._dof, h)
    return hash(dsm._mass, h)
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
    return DensitySampleMeasure(new_smpls, dof = dsm._dof, ess = dsm._ess, mass = dsm._mass)
end

# Disambiguates against unshaped(x, ::ConstValueShape) of ValueShapes:
ValueShapes.unshaped(dsm::DensitySampleMeasure, vs::ConstValueShape) =
    invoke(unshaped, Tuple{DensitySampleMeasure,AbstractValueShape}, dsm, vs)

@inline samplesof(dsm::DensitySampleMeasure) = dsm._smpls


function _with_owner_sampling_law(
    smpls::DensitySampleVector,
    owner::DensitySampleMeasure;
    dof = owner._dof,
    ess = owner._ess,
    mass = owner._mass,
)
    owner_smpls = samplesof(owner)
    owned_smpls = smpls.weight === owner_smpls.weight ? smpls :
        DensitySampleVector((smpls.v, smpls.logd, owner_smpls.weight, smpls.info, smpls.aux))
    return DensitySampleMeasure(owned_smpls, owner._cumulative_weight, dof, ess, mass)
end

# Empirical representations of one pushforward share one owned sampling law.
function _with_sample_weights(dsm::DensitySampleMeasure, owner::DensitySampleMeasure)
    smpls = samplesof(dsm)
    owner_smpls = samplesof(owner)
    smpls.weight === owner_smpls.weight && dsm._cumulative_weight === owner._cumulative_weight && return dsm
    return _with_owner_sampling_law(smpls, owner; dof = dsm._dof, ess = dsm._ess, mass = dsm._mass)
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

function _weight_cdf(W::AbstractVector{<:Real})
    isempty(W) && return similar(W, _weight_accum_type(W))
    rel_weights = _canonical_rel_weights(W)
    cdf = similar(rel_weights, _weight_accum_type(rel_weights))
    copyto!(cdf, rel_weights)
    return cumsum!(cdf, cdf)
end

function _rand_subsample_idx(gen::GenContext, dsm::DensitySampleMeasure)
    # TODO: Use PSIS.

    CW = dsm._cumulative_weight
    isempty(CW) && throw(ArgumentError("Can't draw from an empty DensitySampleMeasure"))
    r = rand(get_rng(gen)) * CW[end]
    idx = _weight_cdf_idx(CW, r)
    return idx
end

@inline function _weight_cdf_idx(CW::AbstractVector, r::Real)
    idx = searchsortedlast(CW, r) + 1
    return idx <= lastindex(CW) ? idx : searchsortedfirst(CW, r)
end

function _rand_subsample_idxs(gen::GenContext, dsm::DensitySampleMeasure, n::Integer)
    # TODO: Use PSIS.

    iszero(n) && return Int[]
    CW = dsm._cumulative_weight
    isempty(CW) && throw(ArgumentError("Can't draw from an empty DensitySampleMeasure"))
    # Always generate R on CPU for now:
    R = rand(get_rng(gen), n) .* CW[end]
    idxs = _weight_cdf_idx.(Ref(CW), R)
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
    return DensitySampleMeasure(new_smpls, dsm._cumulative_weight, dsm._dof, dsm._ess, new_mass)
end


# Multinomial resampling destroys the process order, so per-sample
# process provenance (MCMC sample ids) must not survive it - unlike
# order-preserving systematic resampling, which keeps its ids:
function _without_sampleids(dsm::DensitySampleMeasure)
    s = samplesof(dsm)
    new_s = DensitySampleVector((s.v, s.logd, s.weight, fill(nothing, length(eachindex(s))), s.aux))
    return DensitySampleMeasure(new_s, dsm._cumulative_weight, dsm._dof, dsm._ess, dsm._mass)
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
        _unweighted_resampling_byidxs(p.transformed, resampled_idxs; preserve_ess, owner = main)
    BispacedMeasure(main, transformed, p.f_hash)
end

function _unweighted_resampling_byidxs(
    dsm::DensitySampleMeasure,
    resampled_idxs::AbstractVector{<:Integer};
    preserve_ess::Bool = false,
    owner::Union{Nothing,DensitySampleMeasure} = nothing,
)
    smpls = samplesof(dsm)
    # Resampling replaces the sampling law, so only index payload columns.
    # The main side creates its one owned uniform law; a paired transformed
    # side adopts that exact law without materializing a source-weight column.
    weights = isnothing(owner) ?
        ones(eltype(smpls.weight), length(resampled_idxs)) :
        samplesof(owner).weight
    new_samples = DensitySampleVector((
        smpls.v[resampled_idxs], smpls.logd[resampled_idxs], weights,
        smpls.info[resampled_idxs], smpls.aux[resampled_idxs],
    ))
    old_ess = getess(dsm)
    # Random resampling adds conditional Monte Carlo variance, giving the
    # approximate effective count below. Identity systematic resampling
    # adds none:
    n_new = length(new_samples)
    new_ess = preserve_ess || isnothing(old_ess) ? old_ess : old_ess * n_new / (n_new + old_ess)
    return isnothing(owner) ?
        DensitySampleMeasure(new_samples, _weight_cdf(weights), dsm._dof, new_ess, massof(dsm)) :
        _with_owner_sampling_law(new_samples, owner; dof = dsm._dof, ess = new_ess, mass = massof(dsm))
end
