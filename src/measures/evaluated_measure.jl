# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BAT.MeasureEvalInfo

*Experimental feature, not part of stable public API yet.*

Properties:

* `algorithm`: The algorithm used to evaluate the measure.
* `result`: Algorithm-specific evaluation result.
"""
struct MeasureEvalInfo{Alg,R}
    algorithm::Alg
    result::R
    # ToDo: Store original evaluation context?
    # context::Ctx # Ctx<:BATContext
end


"""
    abstract type AbstractSampleGenerator

*BAT-internal, not part of stable public API.*

Abstract super type for sample generators.
"""
abstract type AbstractSampleGenerator end
export AbstractSampleGenerator


function LazyReports.pushcontent!(rpt::LazyReport, generator::AbstractSampleGenerator)
    alg = getproposal(generator)
    if !(isnothing(alg) || ismissing(alg))
        lazyreport!(rpt, """
        ### Sample generation:

        * Algorithm: $(nameof(typeof(alg)))
        """)
    end
end


"""
    struct EvaluatedMeasure <: BATMeasure

Combines a measure with samples and other information on it.

Constructors:

```julia
em = EvaluatedMeasure(
    measure;
    empirical = ..., approx = ..., dof = ..., mass = ..., modes = ...,
    samplegen = ..., evalinfo = ...
)

BAT.unevaluated(em) === measure
```

[`unevaluated(em)`](@ref) returns the original `measure`.

If `measure` is itself an `EvaluatedMeasure`, the keyword arguments update
its content: given values replace the corresponding entries,
`ScopedSettings.unchanged` (the default) keeps them, and `nothing` (resp.
`MeasureBase.UnknownMass()` for `mass`) clears them. This also provides
directional merging of two evaluated measures, e.g.
`EvaluatedMeasure(em1, approx = em2.approx)`. Use `Accessors.@set` for
direct field surgery that bypasses the update logic.

Properties:

* `unevaluated`: The original measure.
* `empirical`: A [`DensitySampleMeasure`](@ref) based on samples drawn from
  the measure, or `nothing` if no samples are available.
* `approx`: An approximation of the measure, or `nothing` if no approximation
  is available.
* `dof`: The degrees of freedom of the measure, or `nothing` if unknown.
* `mass`: The mass of the measure, or a `MeasureBase.AbstractUnknownMass` if
  unknown.
* `modes`: The modes of the measure, or `nothing` if unknown.
* `samplegen`: An object that carries the necessary information to generate
  samples, the contents is algorithm-specific and not part of the stable API.
  May be `nothing` if no sample generation scheme has been computed.
* `evalinfo`: Information on the (last) evaluation step that
  generated/updated this measure, or `nothing` if no evaluation has been
  performed or information on it is not available.
"""
struct EvaluatedMeasure{
    M<:BATMeasure,
    S<:Union{DensitySampleMeasure,Nothing},
    A<:Union{BATMeasure,Nothing},
    N<:Union{IntegerLike,Nothing},
    U<:Union{Real,MeasureBase.AbstractUnknownMass},
    P<:Union{AbstractVector,Nothing},
    G<:Union{AbstractSampleGenerator,Nothing},
} <: BATMeasure
    unevaluated::M
    empirical::S
    approx::A
    dof::N
    mass::U
    modes::P
    samplegen::G
    evalinfo::Union{MeasureEvalInfo,Nothing}
end
export EvaluatedMeasure


Base.convert(::Type{EvaluatedMeasure}, em::EvaluatedMeasure) = em

function Base.convert(::Type{EvaluatedMeasure}, measurelike::MeasureLike)
    m = batmeasure(measurelike)
    return EvaluatedMeasure(
        m,
        empiricalof(m),
        approxof(m),
        _dofval_or_nothing(getdof(m)),
        massof(m),
        maybe_modes(m),
        nothing, # ToDo: maybe_samplegen(m) or similar
        nothing
    )
end


function EvaluatedMeasure(
    measurelike::MeasureLike;
    empirical::Union{DensitySampleMeasure,DensitySampleVector,Nothing,Unchanged} = unchanged,
    approx::Union{BATMeasure,Nothing,Unchanged} = unchanged,
    dof::Union{IntegerLike,MeasureBase.NoDOF,Nothing,Unchanged} = unchanged,
    mass::Union{RealLike,MeasureBase.AbstractUnknownMass,Unchanged} = unchanged,
    modes::Union{AbstractVector,Nothing,Unchanged} = unchanged,
    samplegen::Union{AbstractSampleGenerator,Nothing,Unchanged} = unchanged,
    evalinfo::Union{MeasureEvalInfo,Nothing,Unchanged} = unchanged
)
    em = convert(EvaluatedMeasure, measurelike)

    if (
        empirical isa Unchanged && approx isa Unchanged && dof isa Unchanged &&
        mass isa Unchanged && modes isa Unchanged &&
        samplegen isa Unchanged && evalinfo isa Unchanged
    )
        return em
    end

    new_empirical = empirical isa Unchanged ? em.empirical :
        isnothing(empirical) ? nothing : convert(DensitySampleMeasure, empirical)
    new_approx = approx isa Unchanged ? em.approx : approx

    new_dof = if dof isa Unchanged
        choose_something(
            _getdof_or_nothing(em),
            _getdof_or_nothing(new_empirical),
            _getdof_or_nothing(new_approx),
        )
    else
        _dofval_or_nothing(dof)
    end

    new_mass = mass isa Unchanged ? _getmass_or_unkown(em) : mass

    # ToDo: Set DOF in empirical if not there yet and inferrable from em.unevaluated?

    return EvaluatedMeasure(
        em.unevaluated,
        new_empirical,
        new_approx,
        new_dof,
        new_mass,
        modes isa Unchanged ? em.modes : modes,
        samplegen isa Unchanged ? em.samplegen : samplegen,
        evalinfo isa Unchanged ? em.evalinfo : evalinfo
    )
end

_getdof_or_nothing(::Nothing) = nothing
_getdof_or_nothing(measure::BATMeasure) = _dofval_or_nothing(getdof(measure))

_dofval_or_nothing(::Nothing) = nothing
_dofval_or_nothing(dof::IntegerLike) = dof
_dofval_or_nothing(::MeasureBase.NoDOF) = nothing
_dofval_or_nothing(dof) = throw(ArgumentError("Degrees of freedom must be an integer or MeasureBase.NoDOF, not $(nameof(typeof(dof)))."))

_getmass_or_unkown(::Nothing) = MeasureBase.UnknownMass()
_getmass_or_unkown(measure::BATMeasure) = massof(measure)


@inline unevaluated(em::EvaluatedMeasure) = em.unevaluated

function empiricalof(em::EvaluatedMeasure)
    if isnothing(em.empirical) && (em.unevaluated isa DensitySampleMeasure)
        return em.unevaluated
    else
        return em.empirical
    end
end

function samplesof(em::EvaluatedMeasure)
    dsm = empiricalof(em)
    return isnothing(dsm) ? nothing : samplesof(dsm)
end

@inline approxof(em::EvaluatedMeasure) = em.approx
MeasureBase.getdof(em::EvaluatedMeasure) = something(em.dof, MeasureBase.NoDOF{typeof(unevaluated(em))}())
MeasureBase.massof(em::EvaluatedMeasure) = em.mass
maybe_modes(em::EvaluatedMeasure) = em.modes
getess(em::EvaluatedMeasure) = getess(_empirical_or_unevaluated(em))
@inline evalinfo(em::EvaluatedMeasure) = em.evalinfo

_evalresult_nt(obj) = _evalresult_nt(evalinfo(obj))
_evalresult_nt(::Nothing) = (;)
_evalresult_nt(info::MeasureEvalInfo) = info.result

# ToDo: Accessors support for empiricalof, approxof, massof, modes, evalinfo and "maybe_samplegen".

# ToDo: How to name this better?
@inline maybe_samplegen(em::EvaluatedMeasure) = em.samplegen

function StatsBase.modes(em::EvaluatedMeasure)
    em_modes = maybe_modes(em)
    if isnothing(em_modes)
        throw(ArgumentError("No mode information available for EvaluatedMeasure"))
    else
        return em_modes
    end
end

function StatsBase.mode(em::EvaluatedMeasure)
    em_modes = modes(em)
    if length(em_modes) > 1
        throw(ArgumentError("EvaluatedMeasure has multiple modes"))
    else
        return only(em_modes)
    end
end

function DensitySampleVector(em::EvaluatedMeasure)
    dsm = empiricalof(em)
    if isnothing(dsm)
        throw(ArgumentError("EvaluatedMeasure has no empirical samples attached to it."))
    else
        return DensitySampleVector(dsm)
    end
end
Base.convert(::Type{DensitySampleVector}, em::EvaluatedMeasure) = DensitySampleVector(em)


Base.showable(::MIME"text/plain", ::EvaluatedMeasure) = true
Base.show(io::IO, mime::MIME"text/plain", em::EvaluatedMeasure) = _show_evaluated_measure(io, mime, em)

Base.showable(::MIME"text/html", ::EvaluatedMeasure) = true
Base.show(io::IO, mime::MIME"text/html", em::EvaluatedMeasure) = _show_evaluated_measure(io, mime, em)

# ToDo: Support ::MIME"juliavscode/html" ?
# Base.showable(::MIME"juliavscode/html", ::EvaluatedMeasure) = true
# Base.show(io::IO, mime::MIME"juliavscode/html", em::EvaluatedMeasure) = _show_evaluated_measure(io, mime, em)

function Base.show(io::IO, em::EvaluatedMeasure)
    print(io, "EvaluatedMeasure(")
    show(io, unevaluated(em))
    print(io, "; ...)")
end

function _show_evaluated_measure(@nospecialize(io::IO), @nospecialize(mime::MIME), @nospecialize(em::EvaluatedMeasure))
    smpls = samplesof(em)
    if get(io, :compact, false) || isnothing(smpls)
        show(io, em)
    else
        rpt = lazyreport()
        lazyreport!(rpt, em)
        show(io, mime, rpt)
    end
end


DensityInterface.logdensityof(em::EvaluatedMeasure, v::Any) = logdensityof(unevaluated(em), v)
DensityInterface.logdensityof(em::EvaluatedMeasure) = logdensityof(unevaluated(em))


ValueShapes.varshape(em::EvaluatedMeasure) = varshape(em.unevaluated)

# `unshaped` is a pure reparametrization, so all measure knowledge is
# transported to the unshaped space. Use `unevaluated` to strip the knowledge
# and obtain a bare measure for performance-critical density evaluation:
function ValueShapes.unshaped(em::EvaluatedMeasure, vs::AbstractValueShape)
    new_measure = unshaped(em.unevaluated, vs)
    new_empirical = isnothing(em.empirical) ? nothing : unshaped(em.empirical, vs)
    new_approx = isnothing(em.approx) ? nothing : unshaped(em.approx, vs)
    new_modes = isnothing(em.modes) ? nothing : unshaped.(em.modes, Ref(vs))
    return EvaluatedMeasure(
        new_measure, new_empirical, new_approx, em.dof, em.mass,
        new_modes, em.samplegen, em.evalinfo
    )
end


has_uhc_support(em::EvaluatedMeasure) = has_uhc_support(em.unevaluated)


# ToDo: truncate_batmeasure(em::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

function MeasureBase.weightedmeasure(logweight::Real, em::EvaluatedMeasure)
    # ToDo: Should approx be reweighted here instead of being kept as-is?
    new_measure = weightedmeasure(logweight, em.unevaluated)
    new_empirical = _renormalize_empirical_logd(logweight, empiricalof(em))
    new_mass = _reweighted_mass(logweight, em.mass)
    return EvaluatedMeasure(
        new_measure, new_empirical, em.approx, em.dof, new_mass,
        em.modes, em.samplegen, nothing
    )
end


function LazyReports.pushcontent!(rpt::LazyReport, em::EvaluatedMeasure)
    smpls = samplesof(em)
    isnothing(smpls) || lazyreport!(rpt, smpls)
    samplegen = maybe_samplegen(em)
    isnothing(samplegen) || lazyreport!(rpt, samplegen)
    return nothing
end


function _empirical_or_unevaluated(em::EvaluatedMeasure)
    empirical = empiricalof(em)
    return !isnothing(empirical) ? empirical : unevaluated(em)
end


Statistics.mean(em::EvaluatedMeasure) = mean(_empirical_or_unevaluated(em))
Statistics.median(em::EvaluatedMeasure) = median(_empirical_or_unevaluated(em))
Statistics.var(em::EvaluatedMeasure) = var(_empirical_or_unevaluated(em))
Statistics.std(em::EvaluatedMeasure) = std(_empirical_or_unevaluated(em))
Statistics.cov(em::EvaluatedMeasure) = cov(_empirical_or_unevaluated(em))

_approx_mean(em::EvaluatedMeasure, n) = _approx_mean(_empirical_or_unevaluated(em), n)
_approx_cov(em::EvaluatedMeasure, n) = _approx_cov(_empirical_or_unevaluated(em), n)


function _approx_max_logd(em::EvaluatedMeasure)
    max_logd = _approx_max_logd(_empirical_or_unevaluated(em))
    @assert ismissing(max_logd) || !isnan(max_logd)
    return max_logd
end
