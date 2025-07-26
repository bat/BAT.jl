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

Combined a measure with samples, and other information on it.

Constructors:

```julia
em = EvaluatedMeasure(
    measure;
    samples = ..., empirical = ..., mass = ..., mode = ...,
    samplegen = ...
)

BAT.unevaluated(em) === measure
```

[`unevaluated(em)`](@ref) returns the original `measure`.

Properties:

* `measure`: The original measure.
* `empirical`: Samples drawn from the measure, or `nothing` if no samples are available.
* `approx`: An approximation of the measure, or `nothing` if no approximation is available
* `dof`: The degrees of freedom of the measure, or `nothing` if unknown.
* `mass`: The mass of the measure, or a `MeasureBase.AbstractUnknownMass` if unknown.
* `modes`: The modes of the measure, or `nothing` if unknown.
* `samplegen`: An object that carries the necessary information to generate samples, the
   contents is algorithm-specific and not part of the stable APT. May be `nothing` if no
   sample generation scheme has been computed.
* `evalinfo`: Information on the (last) evaluation step that generated/updated
  this measure, or `nothing` if no evaluation has been performed or information on it is
  not available.
"""
struct EvaluatedMeasure{
    M<:BATMeasure,
    S<:Union{DensitySampleMeasure,Nothing},
    A<:Union{BATMeasure,Nothing},
    N<:Union{IntegerLike,Nothing},
    U<:Union{Real,Nothing},
    P<:Union{AbstractVector,Nothing},
    G<:Union{AbstractSampleGenerator,Nothing},
    R<:Union{MeasureEvalInfo,Nothing},
} <: BATMeasure
    unevaluated::M
    empirical::S
    approx::A
    dof::N
    mass::U
    modes::P
    samplegen::G
    evalinfo::R
end
export EvaluatedMeasure


Base.convert(::Type{EvaluatedMeasure}, em::EvaluatedMeasure) = em

function Base.convert(::Type{EvaluatedMeasure}, measurelike::MeasureLike)
    return EvaluatedMeasure(
        measure,
        empiricalof(measure),
        approxof(measure),
        getdof(measure),
        massof(measure),
        maybe_modes(measure),
        nothing, # ToDo: maybe_samplegen(measure) or similar
        evalinfo(measure)
    )
end


function EvaluatedMeasure(
    measurelike::MeasureLike;
    empirical::Union{DensitySampleMeasure,Nothing} = nothing,
    approx::Union{BATMeasure,Nothing} = nothing,
    dof::Union{IntegerLike,MeasureBase.NoDOF,Nothing} = nothing,
    mass::Union{RealLike,MeasureBase.AbstractUnknownMass} = MeasureBase.UnknownMass(),
    modes::Union{AbstractVector,Nothing} = nothing,
    samplegen::Union{AbstractSampleGenerator,Nothing} = nothing,
    evalinfo::Union{MeasureEvalInfo,Nothing} = nothing
)
    em = convert(EvaluatedMeasure, measurelike)

    new_dof = choose_something(
        _dofval_or_nothing(dof),
        _getdof_or_nothing(em),
        _getdof_or_nothing(empirical),
        _getdof_or_nothing(approx),
    )

    new_mass = choose_something(
        mass,
        _getmass_or_unkown(em),
    )

    # If nothing has changed, return em directly:
    if (
        isnothing(empirical) && isnothing(approx) && (mass isa MeasureBase.UnknownMass) &&
        isnothing(modes) && isnothing(samplegen) && isnothing(evalinfo)
    )
        return em
    end

    # ToDo: Set DOF in empirical if not there yet and inferrable from em.unevaluated?

    return EvaluatedMeasure(
        em.unevaluated,
        choose_something(empirical, em.empirical),
        choose_something(approx, em.approx),
        new_dof,
        new_mass,
        choose_something(modes, em.modes),
        choose_something(samplegen, em.samplegen),
        evalinfo # never keep old evalinfo if anything has changed
    )
end

_getdof_or_nothing(::Nothing) = nothing
_getdof_or_nothing(measure::BATMeasure) = _dofval_or_nothing(getdof(measure))

_dofval_or_nothing(dof::IntegerLike) = dof
_dofval_or_nothing(::MeasureBase.NoDOF) = nothing
_dofval_or_nothing(dof) = throw(ArgumentError("Degrees of freedom must be an integer of MeasureBase.NoDOf, not $(nameof(typeof(dof)))."))

_getmass_or_unkown(::Nothing) = MeasureBase.UnknownMass()
_getmass_or_unkown(measure::BATMeasure) = getmass(measure)


@inline unevaluated(em::EvaluatedMeasure) = em.unevaluated

function empiricalof(em::EvaluatedMeasure)
    if isnothing(em.empirical) && (em.unevaluated isa DensitySampleMeasure)
        return em.unevaluated
    else
        return em.empirical
    end
end

@inline samplesof(em::EvaluatedMeasure) = empiricalof(em.empirical) isa Nothing ? nothing : samplesof(empiricalof(em))
@inline approxof(em::EvaluatedMeasure) = em.approx
MeasureBase.getdof(em::EvaluatedMeasure) = something(em._dof, MeasureBase.NoDOF{typeof(unevaluated(em))}())
MeasureBase.massof(em::EvaluatedMeasure) = em.mass
maybe_modes(em::EvaluatedMeasure) = em.modes
@inline evalinfo(em::EvaluatedMeasure) = em.evalinfo

StatsBase.modes(em::EvaluatedMeasure) = something(maybe_modes(em.modes))

# ToDo: Accessors support for empiricalof, approxof, massof, modes, evalinfo and "maybe_samplegen".

# ToDo: How to name this better?
@inline maybe_samplegen(em::EvaluatedMeasure) = em.samplegen

function StatsBase.mode(em::EvaluatedMeasure)
    em_modes = modes(em)
    if isnothing(em_modes)
        return nothing
    elseif length(em_modes) > 1
        throw(ArgumentError("EvaluatedMeasure of type $(nameof(typeof(em))) has multiple modes"))
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

function _show_evaluated_measure(@nospecialize(io::IO), @nospecialize(mime::MIME), @nospecialize(em::EvaluatedMeasure))
    smpls = samplesof(em)

    if get(io, :compact, false) || !isnothing(smpls)
        print(io, "EvaluatedMeasure(")
        show(io, unevaluated(em))
        print(io, "; ...)")
    else
        buf = IOBuffer()
        tmpio  = IOContext(buf, :compact => true)
        show(tmpio, em)
        em_str = String(take!(buf))
        rpt = lazyreport()
        smpls = samplesof(em)
        if isnothing(smpls)
            show(io,em_str)
        else
            lazyreport!(rpt, "$em_str with samples")
            lazyreport!(rpt, smpls, "$em_str with samples")
        end
        show(io, mime, rpt)
    end
end


DensityInterface.logdensityof(em::EvaluatedMeasure, v::Any) = logdensityof(unevaluated(em), v)
DensityInterface.logdensityof(em::EvaluatedMeasure) = logdensityof(unevaluated(em))


ValueShapes.varshape(em::EvaluatedMeasure) = varshape(em.unevaluated)

function ValueShapes.unshaped(em::EvaluatedMeasure, vs::AbstractValueShape)
    new_measure = unshaped(em.unevaluated, vs)
    @assert varshape(em.empirical) <= vs
    new_sampled = unshaped(em.empirical)
    return EvaluatedMeasure(new_measure, new_sampled, em.approx, em.mass, em.modes, em.samplegen)
end


measure_support(em::EvaluatedMeasure) = measure_support(em.unevaluated)


# ToDo: truncate_batmeasure(em::EvaluatedMeasure, bounds::AbstractArray{<:Interval})

function MeasureBase.weightedmeasure(logweight::Real, em::EvaluatedMeasure)
    new_measure = weightedmeasure(logweight, em.unevaluated)
    new_empirical = _renormalize_empirical_logd(logweight, empiricalof(em))
    return EvaluatedMeasure(new_measure, new_empirical, em.approx, em.mass, em.modes, em.samplegen)
end


function LazyReports.pushcontent!(rpt::LazyReport, em::EvaluatedMeasure)
    lazyreport!(rpt, samplesof(em))
    lazyreport!(rpt, maybe_samplegen(em))
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
    _max_logd = _approx_max_logd(_empirical_or_unevaluated(em))
    @assert !isnan(_max_logd)
    return _max_logd
end
