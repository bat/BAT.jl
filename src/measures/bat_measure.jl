# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.unevaluated(obj)

If `obj` is an evaluated object, like a [`EvaluatedMeasure`](@ref),
return the original (unevaluated) object. Otherwise, return `obj`.

This is the explicit way to strip attached measure knowledge, e.g. to
obtain a bare measure for performance-critical density evaluation.
Reparametrizations like `unshaped` transport attached knowledge instead
of dropping it.
"""
function unevaluated end
export unevaluated

unevaluated(obj) = obj


"""
    empiricalof(m)::Union{DensitySampleMeasure,Nothing}

Get the empirical measure, based on samples drawn from measure-like object
`m`, associated with `m`, or `nothing` if no empirical representation is
available. Also see [`EvaluatedMeasure`](@ref).
"""
function empiricalof end
export empiricalof

empiricalof(::AbstractMeasure) = nothing

# Like `empiricalof`, but returns a BispacedMeasure pair:
_empirical_rep(m::AbstractMeasure) = _as_bispaced(empiricalof(m))

"""
    samplesof(m)::Union{DensitySampleVector,Nothing}

Get the samples associated with measure-like object `m`, or `nothing` if
no samples are available.

The returned object is live internal data of `m`, it must not be modified.
Use `DensitySampleVector(m)` or `convert(DensitySampleVector, m)` to obtain
an independent copy from a `DensitySampleMeasure` or an `EvaluatedMeasure`
with empirical samples.
"""
function samplesof end
export samplesof

samplesof(::AbstractMeasure) = nothing

"""
    approxof(m)::Union{AbstractMeasure,Nothing}

Get an approximation of measure-like object `m`, or `nothing` if no
approximation is available.
"""
function approxof end
export approxof

approxof(::AbstractMeasure) = nothing

"""
    samplegenof(m)::Union{BAT.AbstractSampleGenerator,Nothing}

Get the sample generation scheme associated with measure-like object `m`,
or `nothing` if none has been computed. The contents of sample generators
is algorithm-specific and not part of the stable API.
"""
function samplegenof end
export samplegenof

samplegenof(::AbstractMeasure) = nothing

"""
    getess(m)::Union{Real,Nothing}

Get the (scalar) effective sample size associated with measure-like object
`m`, or `nothing` if unknown.
"""
function getess end
export getess

getess(::AbstractMeasure) = nothing

"""
    evalinfo(m)::Union{BAT.MeasureEvalInfo,Nothing}

Get information on the (last) evaluation step that generated or updated
measure-like object `m`, or `nothing` if no such information is available.
The contents of evaluation information is algorithm-specific and not part
of the stable API.
"""
function evalinfo end
export evalinfo

evalinfo(::AbstractMeasure) = nothing

maybe_modes(::AbstractMeasure) = nothing

function some_dof(m::AbstractMeasure)
    n_dof = getdof(m)
    if n_dof isa MeasureBase.NoDOF
        throw(ArgumentError("Can't determine degrees of freedom for measure of type $(nameof(typeof(m)))"))
    else
        return n_dof
    end
end


function _rv_dof(m::AbstractMeasure)
    tv = testvalue(m)
    if !(tv isa AbstractVector{<:Real})
        throw(ArgumentError("Measure of type $(nameof(typeof(m))) is not on the space of real-valued vectors"))
    end
    length(eachindex(tv))
end


"""
    batmeasure(obj)

*Experimental feature, not part of stable public API.*

Convert a measure-like `obj` to a measure that is compatible with BAT.

`batmeasure` is BAT's canonicalization. It is idempotent, turns
distributions into measures via `MeasureBase.asmeasure`, named tuples of
distributions into product measures, and the density measures of
MeasureBase (e.g. `mintegrate_exp(likelihood, prior)`) into
[`PosteriorMeasure`](@ref)s.
"""
function batmeasure end
export batmeasure

batmeasure(obj) = asmeasure(obj)
batmeasure(::Missing) = missing

batmeasure(ds::NamedTuple) = productmeasure(map(_marginal_measure, ds))

# A density measure over a prior is a (possibly nested) posterior measure:
batmeasure(m::DensityMeasure) = PosteriorMeasure(m)


"""
    supports_rand(m)

*BAT-internal, not part of stable public API.*

Check whether a measure-like object `m` supports `rand`.
"""
@inline supports_rand(::AbstractMeasure) = true
@inline supports_rand(m::WeightedMeasure) = supports_rand(m.base)
@inline supports_rand(m::PushforwardMeasure) = !(gettransform(m) isa NoInverse) && supports_rand(m.origin)


"""
    has_uhc_support(m)::Bool

*BAT-internal, not part of stable public API.*

Is the support of measure `m` limited to the unit hypercube?
"""
has_uhc_support(::AbstractMeasure) = false
has_uhc_support(::StdUniform) = true
has_uhc_support(m::PowerMeasure) = has_uhc_support(_pwr_base(m))
has_uhc_support(m::WeightedMeasure) = has_uhc_support(m.base)
has_uhc_support(m::MeasureBase.AbstractProductMeasure) = all(has_uhc_support, marginals(m))
has_uhc_support(m::AsMeasure) = has_uhc_support(m.obj)

# A pushforward lives where its transformation maps to. Transformations
# that only change the variate shape pass the question on (`missing`), so
# that the last step that determines the variate values decides:
has_uhc_support(m::PushforwardMeasure) = _uhc_support_from(_maps_to_uhc(gettransform(m)), m)
_uhc_support_from(known::Bool, ::PushforwardMeasure) = known
_uhc_support_from(::Missing, m::PushforwardMeasure) = has_uhc_support(m.origin)

_maps_to_uhc(::Any) = false
_maps_to_uhc(f::TransportFunction) = has_uhc_support(f.ν)
_maps_to_uhc(f::FunctionChain) = _chain_maps_to_uhc(fchainfs(f))
_maps_to_uhc(f::ComposedFunction) = _chain_maps_to_uhc((f.inner, f.outer))

function _chain_maps_to_uhc(fs::Tuple)
    r = _maps_to_uhc(last(fs))
    return ismissing(r) && length(fs) > 1 ? _chain_maps_to_uhc(Base.front(fs)) : r
end

has_uhc_support(::Distribution) = false
has_uhc_support(d::Distribution{Univariate,Continuous}) = minimum(d) ≈ false && maximum(d) ≈ true
has_uhc_support(d::ReshapedDist) = has_uhc_support(unshaped(d))

"""
    BAT.is_std_mvnormal(m)::Bool

*BAT-internal, not part of stable public API.*

Is `m` a standard multivariate normal measure?
"""
is_std_mvnormal(::AbstractMeasure) = false
is_std_mvnormal(m::PowerMeasure) = _pwr_base(m) isa StdNormal && length(_pwr_size(m)) == 1
is_std_mvnormal(m::AsMeasure) = is_std_mvnormal(m.obj)

is_std_mvnormal(::Distribution) = false
is_std_mvnormal(d::MvNormal) = mean(d) ≈ Zeros(length(d)) && cov(d) ≈ I(length(d))



# Moments and modes of wrapped distributions are known analytically:

const _DistMeasure = AsMeasure{<:Distribution}

Statistics.mean(m::_DistMeasure) = mean(m.obj)
Statistics.median(m::AsMeasure{<:UnivariateDistribution}) = median(m.obj)
Statistics.var(m::_DistMeasure) = var(m.obj)
Statistics.std(m::_DistMeasure) = std(m.obj)
Statistics.cov(m::AsMeasure{<:MultivariateDistribution}) = cov(m.obj)
StatsBase.mode(m::_DistMeasure) = mode(m.obj)

# Not every distribution has a well-defined mode, failures across
# Distributions.jl surface as varied error types:
function maybe_modes(m::_DistMeasure)
    try
        [mode(m.obj)]
    catch
        nothing
    end
end


# Moments and modes of structural measures follow their structure:

Statistics.mean(m::MeasureBase.AbstractProductMeasure) = map(mean, marginals(m))
Statistics.var(m::MeasureBase.AbstractProductMeasure) = map(var, marginals(m))
StatsBase.mode(m::MeasureBase.AbstractProductMeasure) = map(mode, marginals(m))

Statistics.mean(m::MeasureBase.Dirac) = m.x
Statistics.var(m::MeasureBase.Dirac) = zero.(m.x)
StatsBase.mode(m::MeasureBase.Dirac) = m.x

Statistics.mean(m::WeightedMeasure) = mean(m.base)
Statistics.var(m::WeightedMeasure) = var(m.base)
Statistics.cov(m::WeightedMeasure) = cov(m.base)
StatsBase.mode(m::WeightedMeasure) = mode(m.base)


show_value_shape(io::IO, vs::AbstractValueShape) = show(io, vs)
function show_value_shape(io::IO, vs::NamedTupleShape)
    print(io, Base.typename(typeof(vs)).name, "(")
    show(io, propertynames(vs))
    print(io, "}(…)")
end


function _reweighted_mass(logweight::Real, current_mass::Real)
    current_logmass = _lfloat(log(asnonstatic(current_mass)))
    new_logmass = oftype(current_logmass, logweight) + current_logmass
    return exp(ULogarithmic, new_logmass)
end

_reweighted_mass(::Real, current_mass::MeasureBase.AbstractUnknownMass) = current_mass


# ToDo: This should just be a method of a proper `bat_renormalize` API function
# when using an `AutoRenormalize` (or similar name) algorithm:
"""
    BAT.auto_renormalize(measure::MeasureBase.AbstractMeasure)

*Experimental feature, not part of stable public API.*

Returns `(result = new_measure, logweight = logweight)`.

Tries to automatically renormalize `measure` if a maximum log-density value
is available, returns `measure` unchanged otherwise.
"""
function auto_renormalize(measure::AbstractMeasure)
    _generic_auto_renormalize_impl(_approx_max_logd(measure), batmeasure(measure))
end


_approx_max_logd(::AbstractMeasure) = missing
_approx_max_logd(::Nothing) = missing

function _approx_max_logd(samples::DensitySampleVector)
    logweight = maximum(samples.logd)
    isnan(logweight) || isinf(logweight) ? zero(logweight) : logweight
end

function _generic_auto_renormalize_impl(max_logd::Real, measure::AbstractMeasure)
    logweight = - max_logd
    result = _bat_weightedmeasure(logweight, measure)
    (result = result, logweight = logweight)
end

function _generic_auto_renormalize_impl(::Missing, measure::AbstractMeasure)
    (result = measure, logweight = false)
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
