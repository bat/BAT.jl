# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BAT.LFDensity{F}

*BAT-internal, not part of stable public API.*

Wraps a log-density function `log_f`.
"""
struct LFDensity{F} <: BATDensity
    _log_f::F
end

Base.convert(::Type{LFDensity}, density::DensityInterface.LogFuncDensity) = LFDensity(logdensityof(density))

@inline DensityInterface.logdensityof(density::LFDensity, x) = density._log_f(x)
@inline DensityInterface.logdensityof(density::LFDensity) = density._log_f

function Base.show(io::IO, density::LFDensity)
    print(io, Base.typename(typeof(density)).name, "(")
    show(io, density._log_f)
    print(io, ")")
end

_precompose_density(density::LFDensity, g) = LFDensity(ffcomp(density._log_f, g))



"""
    BAT.LFDensityWithGrad{F,G} <: BATDensity

*BAT-internal, not part of stable public API.*

Constructors:

    LFDensityWithGrad(logf, valgradlogf)

A density defined by a function that computes it's logarithmic value at given
points, as well as a function that computes both the value and the gradient.

It must be safe to execute both functions in parallel on multiple threads and
processes.
"""
struct LFDensityWithGrad{F,G} <: BATDensity
    logf::F
    valgradlogf::G
end

DensityInterface.logdensityof(density::LFDensityWithGrad) = density.logf

function DensityInterface.logdensityof(density::LFDensityWithGrad, v::Any)
    density.logf(v)
end

function ChainRulesCore.rrule(::typeof(DensityInterface.logdensityof), density::LFDensityWithGrad, v)
    value, gradient = density.valgradlogf(v)
    @assert value isa Real
    function lfdwg_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        @assert ΔΩ isa Real
        tangent = gradient * ΔΩ
        (NoTangent(), ZeroTangent(), tangent)
    end
    return value, lfdwg_pullback
end


function Base.show(io::IO, density::LFDensityWithGrad)
    print(io, Base.typename(typeof(density)).name, "(")
    show(io, density.logf)
    print(io, ")")
end


"""
    BAT.SupportedDensity(likelihood, support::MeasureBase.AbstractMeasure, f_to_support)

*BAT-internal, not part of stable public API.*

A density that evaluates `likelihood` at `f_to_support(v)`, and is zero
wherever that point lies outside of the support of the measure `support`.

Substituting the prior of a posterior measure (see [`PriorSubstitution`](@ref))
drops the support of the original prior: transports map the boundary of that
support to infinite variates, at which a likelihood need not be defined.
Carrying the original support along with the likelihood keeps the posterior
density zero there instead of undefined.
"""
struct SupportedDensity{D,M<:AbstractMeasure,G} <: BATDensity
    likelihood::D
    support::M
    f_to_support::G
end

SupportedDensity(likelihood, support::AbstractMeasure) = SupportedDensity(likelihood, support, identity)

function DensityInterface.logdensityof(d::SupportedDensity, v)
    x = d.f_to_support(v)
    # `insupport` may be undecidable (`NoFastInsupport`), only a definite
    # `false` masks the likelihood out. The likelihood must not be evaluated
    # in that case, so this is a branch and not an `ifelse`:
    if insupport(d.support, x) == false
        return log_zero_density(realnumtype(typeof(x)))
    else
        return logdensityof(d.likelihood, x)
    end
end

# The likelihood stays in the space of the support, only the map into that
# space is composed:
_precompose_density(d::SupportedDensity, g) = SupportedDensity(d.likelihood, d.support, ffcomp(d.f_to_support, g))

# The forward model of the likelihood includes the map into the support's
# space (it is not masked by the support, as the density is):
_get_model(d::SupportedDensity) = ffcomp(_get_model(d.likelihood), d.f_to_support)
_get_observation(d::SupportedDensity) = _get_observation(d.likelihood)
