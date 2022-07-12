# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    GenericDensity{F<:Function} <: BATDensity

**GenericDensity is deprecated and may be removed in future BAT versions**
"""
struct GenericDensity{F<:Function} <: BATDensity
    f::F

    @noinline function GenericDensity(f::F) where {F<:Function}
        Base.depwarn("`BAT.GenericDensity` is deprecated, use package DensityInterface.jl to turn log-density functions into BAT-compatible densities.", :GenericDensity)
        new{F}(f)
    end    
end


Base.convert(::Type{GenericDensity}, f::Function) = GenericDensity(f)

@noinline function Base.convert(::Type{AbstractMeasureOrDensity}, f::Function)
    Base.depwarn("`convert(BAT.AbstractMeasureOrDensity, f::Function)` is deprecated, use `convert(AbstractMeasureOrDensity, logfuncdensity(g))` with a function `g` that returns the log-density value directly instead.", :convert)
    GenericDensity(f)
end

Base.parent(density::GenericDensity) = density.f

function DensityInterface.logdensityof(density::GenericDensity, v::Any)
    logvalof(density.f(v))
end



"""
    struct BAT.LFDensity{F}

*BAT-internal, not part of stable public API.*

Wraps a log-density function `log_f`.
"""
struct LFDensity{F} <: BATDensity
    _log_f::F
end

Base.convert(::Type{LFDensity}, density::DensityInterface.LogFuncDensity) = LFDensity(logdensityof(density))
Base.convert(::Type{AbstractMeasureOrDensity}, density::DensityInterface.LogFuncDensity) = convert(LFDensity, density)

@inline DensityInterface.logdensityof(density::LFDensity, x) = density._log_f(x)
@inline DensityInterface.logdensityof(density::LFDensity) = density._log_f

function Base.show(io::IO, density::LFDensity)
    print(io, Base.typename(typeof(density)).name, "(")
    show(io, density._log_f)
    print(io, ")")
end


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

vjp_algorithm(density::LFDensityWithGrad) = ZygoteAD()


function Base.show(io::IO, density::LFDensityWithGrad)
    print(io, Base.typename(typeof(density)).name, "(")
    show(io, density.logf)
    print(io, ")")
end
