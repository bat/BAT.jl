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
    BAT.FiniteVariateDensity(likelihood, f_to_variate)

*BAT-internal, not part of stable public API.*

A density that evaluates `likelihood` at `f_to_variate(v)`, and is zero
wherever that point has non-finite components.

Substituting the prior of a posterior measure (see [`PriorSubstitution`](@ref))
drops the support of the original prior. Transports keep finite inputs
finite, but in the extreme tails they saturate at infinite variates, at
which a likelihood need not be defined. Such points carry no probability
mass, so the density is zero there instead of undefined.
"""
struct FiniteVariateDensity{D,G} <: BATDensity
    likelihood::D
    f_to_variate::G
end

function DensityInterface.logdensityof(d::FiniteVariateDensity, v)
    x = d.f_to_variate(v)
    # The likelihood must not be evaluated at non-finite variates, so this
    # is a branch and not an `ifelse`:
    if _nonfinite_variate(x)
        return log_zero_density(realnumtype(typeof(x)))
    else
        return logdensityof(d.likelihood, x)
    end
end

# The likelihood stays in the variate space, only the map into that space
# is composed:
_precompose_density(d::FiniteVariateDensity, g) = FiniteVariateDensity(d.likelihood, ffcomp(d.f_to_variate, g))

# The forward model of the likelihood includes the map into the variate
# space (it is not masked, as the density is):
_get_model(d::FiniteVariateDensity) = ffcomp(_get_model(d.likelihood), d.f_to_variate)
_get_observation(d::FiniteVariateDensity) = _get_observation(d.likelihood)
