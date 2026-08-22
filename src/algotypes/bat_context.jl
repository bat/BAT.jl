# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct BATContext{T}

*Experimental feature, not yet part of stable public API.*

Set the default computational context for BAT.

Constructors:

```julia
BATContext{T}(rng::AbstractRNG, cunit::AbstractComputeUnit, ADSelector::AD)

BATContext(;
    precision::Type{<:AbstractFloat} = ...,
    rng::AbstractRNG = ...,
    cunit::HeterogeneousComputing.AbstractComputeUnit = ...,
    ad::Union{AutoDiffOperators.ADSelector, Module, Symbol, Val} = ...,
)
```

The default `rng` is seeded from `Random.default_rng()`, so results become
reproducible via `Random.seed!`.

See [`get_batcontext`](@ref), [`set_batcontext`](@ref) and
[`BAT.default_batcontext`](@ref).
"""
struct BATContext{T<:AbstractFloat,RNG<:AbstractRNG,CU<:AbstractComputeUnit,AD<:ADSelector}
    rng::RNG
    cunit::CU
    ad::AD
end

export BATContext

function BATContext{T}(
    rng::RNG, cunit::CU, ad::AD
) where {T<:AbstractFloat,RNG<:AbstractRNG,CU<:AbstractComputeUnit,AD<:ADSelector}
    BATContext{T,RNG,CU,AD}(rng, cunit, ad)
end

function BATContext(;
    precision::Type{T} = Float64,
    rng::AbstractRNG = Philox4x((rand(UInt64), rand(UInt64)))::Philox4x{UInt64,10},
    cunit::AbstractComputeUnit = CPUnit(),
    ad::Union{ADSelector, Module, Symbol, Val} = NoAutoDiff(),
) where T
    adsel = _to_adsel(ad)
    BATContext{T}(rng, cunit, adsel)
end

_to_adsel(ad::ADSelector) = ad
_to_adsel(ad::Module) = ADSelector(ad)
_to_adsel(ad::Symbol) = ADSelector(ad)
_to_adsel(ad::Val) = ADSelector(ad)


HeterogeneousComputing.get_precision(::BATContext{T}) where T = T
HeterogeneousComputing.get_rng(context::BATContext) = context.rng
HeterogeneousComputing.get_compute_unit(context::BATContext) = context.cunit

function HeterogeneousComputing.get_gencontext(context::BATContext)
    GenContext{get_precision(context)}(get_compute_unit(context), get_rng(context))
end


"""
    BAT.get_adselector(context::BATContext)

*Experimental feature, not yet part of stable public API.*

Returns the automatic differentiation selector specified in `context`.
"""
function get_adselector end

get_adselector(context::BATContext) = context.ad



"""
    BAT.get_valid_adselector(context::BATContext, algorithm)

*Experimental feature, not yet part of stable public API.*

Returns the automatic differentiation selector specified in `context`, to
be used for `algorithm`.

Throws an exception if `context` specifies `AutoDiffOperators.NoAutoDiff`.
"""
function get_valid_adselector(context::BATContext, @nospecialize(algorithm))
    ad = get_adselector(context)
    _check_adselector(ad, _algname(algorithm))
    return ad
end

_algname(algname::Symbol) = algname
_algname(algorithm) = nameof(typeof(algorithm))

_check_adselector(::ADSelector, ::Symbol) = nothing

function _check_adselector(::NoAutoDiff, algname::Symbol)
    throw(ErrorException("Algorithm $algname requires automatic differentiation, but no AD backend specified. Pass a BAT context like `BATContext(ad = ForwardDiff)` to the algorithm or set a default `BATContext` with AD e.g. via `set_batcontext(ad = ForwardDiff)`"))
end


"""
    BAT.set_rng(context::BATContext, rng::AbstractRNG)::BATContext

*Experimental feature, not yet part of stable public API.*

Returns a copy of `context` with the random number generator set to `rng`.
"""
function set_rng(context::BATContext{T}, rng::AbstractRNG) where T
    BATContext{T}(rng, get_compute_unit(context), get_adselector(context))
end


function Base.show(io::IO, context::BATContext{T}) where T
    gen = get_gencontext(context)
    print(io, nameof(typeof(context)), "{", T, "}(")
    print(io, get_rng(gen), ", ")
    print(io, get_compute_unit(gen), ", ")
    print(io, get_adselector(context))
    print(io, ")")
end


"""
    BAT.default_batcontext::ScopedSettings.ScopedSetting{BATContext}

*Experimental feature, not yet part of stable public API.*

The setting that holds the default computational context for BAT.

Unless overridden, each access yields a freshly constructed
[`BATContext`](@ref) whose random number generator is seeded from
`Random.default_rng()` (so `Random.seed!` makes BAT results
reproducible).

[`set_batcontext`](@ref) installs a process-wide override, and
`ScopedSettings.default_value` removes it again:

```julia
BAT.default_batcontext[] = default_value
```

A context can also be bound for a dynamic scope only, which is
inherited by tasks started within that scope:

```julia
using ScopedSettings: with

with(BAT.default_batcontext => BATContext(ad = ForwardDiff)) do
    bat_sample(target, MCMCSampling())
end
```

See [`get_batcontext`](@ref) and [`set_batcontext`](@ref).
"""
const default_batcontext = ScopedSetting{BATContext}(BATContext)


"""
    get_batcontext()::BATContext

Gets the default computational context for BAT.

Note: `get_batcontext()` does not have a stable return type. Code that
needs type stability should pass a context to algorithms explicitly.
BAT algorithms that call other algorithms must forward their context
automatically, so context is always type stable within nested
BAT algorithms.

See [`BATContext`](@ref), [`set_batcontext`](@ref) and
[`BAT.default_batcontext`](@ref).
"""
function get_batcontext end
export get_batcontext

get_batcontext() = default_batcontext[]


"""
    set_batcontext(new_context::BATContext)

    set_batcontext(;
        precision = ...,
        rng = ...,
        cunit = ...,
        ad = ...
    )

Sets the default computational context for BAT.

The new context becomes the process-wide default, visible to all tasks.
To override the default for a dynamic scope only, bind
[`BAT.default_batcontext`](@ref) via `ScopedSettings.with` instead -
`set_batcontext` throws when called inside such a scope, as the
scoped binding would shadow the assignment anyway.

See [`BATContext`](@ref) and [`get_batcontext`](@ref).
"""
function set_batcontext end
export set_batcontext

function set_batcontext(context::BATContext)
    default_batcontext[] = context
    return get_batcontext()
end

function set_batcontext(;kwargs...)
    c = get_batcontext()
    s = merge(
        (precision = get_precision(c), rng = get_rng(c), cunit = get_compute_unit(c), ad = get_adselector(c)),
        (;kwargs...)
    )
    adsel = _to_adsel(s.ad)
    set_batcontext(BATContext{s.precision}(s.rng, s.cunit, adsel))
end


const _g_dummy_context = BATContext()
