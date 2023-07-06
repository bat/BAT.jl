# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const _tls_batcontext_key = :_BAT_default_context_

struct _NoADSelected end


"""
    struct BATContext

*Experimental feature, not yet part of stable public API.*

Set the default computational context for BAT.

Constructor:

```julia 
BATContext(;
    precision::Type{<:AbstractFloat} = ...,
    rng::AbstractRNG = ...,
    ad = ...
)
```

See [`default_context`](@ref).
"""
struct BATContext{G<:GenContext,AD<:Union{<:ADSelector,_NoADSelected}}
    _gen::G
    _ad::AD
end
export BATContext

function BATContext(;
    precision::Type{<:AbstractFloat} = Float64,
    rng::AbstractRNG = Philox4x()::Philox4x{UInt64,10},
    ad = _NoADSelected()
)
    BATContext(GenContext{precision}(rng), ad)
end


HeterogeneousComputing.get_gencontext(context::BATContext) = context._gen
HeterogeneousComputing.get_rng(context::BATContext) = get_rng(get_gencontext(context))
HeterogeneousComputing.get_precision(context::BATContext) = get_precision(get_gencontext(context))
HeterogeneousComputing.get_compute_unit(context::BATContext) = get_compute_unit(get_gencontext(context))


"""
    BAT.get_context(obj)::BATContext

*Experimental feature, not yet part of stable public API.*

Returns the context associated with `obj`.
"""
function get_context end


"""
    BAT.set_rng(context::BATContext, rng::AbstractRNG)::BATContext

*Experimental feature, not yet part of stable public API.*

Returns a copy of `context` with the random number generator set to `rng`.
"""
function set_rng(context::BATContext, rng::AbstractRNG)
    BATContext(GenContext{get_precision(context)}(get_compute_unit(context), rng), get_adselector(context))
end


"""
    BAT.get_adselector(context::BATContext)

*Experimental feature, not yet part of stable public API.*

Returns the automatic differentiation selector specified in `context`.
"""
function get_adselector end

get_adselector(context::BATContext) = context._ad


function Base.show(io::IO, context::BATContext)
    gen = get_gencontext(context)
    print(io, nameof(typeof(context)), "(")
    print(io, "precision = ", get_precision(gen), ", ")
    print(io, "rng = ", get_rng(gen), ", ")
    print(io, "ad = ", get_adselector(context))
    print(io, ")")
end


"""
    BAT.default_context()
    BAT.default_context(new_context::BATContext)

*Experimental feature, not yet part of stable public API.*

Gets resp. sets the default computational context for BAT.

Will create and set a new default context if none exists.

Note: `default_context()` does not have a stable return type. Code that
needs type stability should pass a context to algorithms explicitly.
BAT algorithms that call other algorithms must forward their context
automatically, so context is always type stable within nested
BAT algorithms.

See [`BATContext`](@ref).
"""
function default_context()
    if haskey(task_local_storage(), _tls_batcontext_key)
        return task_local_storage(_tls_batcontext_key)
    else
        context = BATContext()
        @info "Setting new default BAT context $context"
        task_local_storage(_tls_batcontext_key, context)
        return context
    end
end

function default_context(context::BATContext)
    task_local_storage(_tls_batcontext_key, context)
    return default_context()
end
