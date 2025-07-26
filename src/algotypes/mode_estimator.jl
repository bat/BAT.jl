# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type BAT.AbstractModeEstimator

Abstract type for BAT optimization algorithms.

A typical application for optimization in BAT is mode estimation
(see [`bat_findmode`](@ref)),
"""
abstract type AbstractModeEstimator end


"""
    bat_findmode(
        target::BAT.AnySampleable,
        [algorithm::BAT.AbstractModeEstimator],
        [context::BATContext]
    )::DensitySampleVector

Estimate the global mode of `target`.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, evaluated::EvaluatedMeasure, ...)
```

(The field `evaluated` is only present if `target` is a measure.)


Result properties not listed here are algorithm-specific and are not part
of the stable public API.

# Implementation

`bat_findmode` uses [`evalmeasure`](@ref) internally. Do not specialize
`bat_findmode`.
"""
function bat_findmode end
export bat_findmode


function bat_findmode(target::AnySampleable, algorithm, context::BATContext)
    orig_context = deepcopy(context)

    em = evalmeasure(target, algorithm, context)
    r = (;result = mode(em), evalinfo(em).result...)

    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_findmode(target::AnySampleable) = bat_findmode(target, get_batcontext())

bat_findmode(target::AnySampleable, algorithm) = bat_findmode(target, algorithm, get_batcontext())

function bat_findmode(target::AnySampleable, context::BATContext)
    algorithm = bat_default_withdebug(bat_findmode, Val(:algorithm), target);
    bat_findmode(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_findmode), ::Val{:algorithm}, x::AbstractModeEstimator)
    "Using mode estimator $x"
end



"""
    bat_marginalmode(
        target::DensitySampleVector,
        algorithm::AbstractModeEstimator,
        [context::BATContext]
    )::DensitySampleVector

*Experimental feature, not part of stable public API.*

Estimates a marginal mode of `target` by finding the maximum of marginalized posterior for each dimension.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```

!!! note

    Do not add add methods to `bat_marginalmode`, add methods to
    `bat_marginalmode_impl` instead.
"""
function bat_marginalmode end
export bat_marginalmode

function bat_marginalmode_impl end


function bat_marginalmode(measure::AnySampleable, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_marginalmode_impl(measure, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_marginalmode(measure::AnySampleable) = bat_marginalmode(measure, get_batcontext())

bat_marginalmode(measure::AnySampleable, algorithm) = bat_marginalmode(measure, algorithm, get_batcontext())

function bat_marginalmode(measure::AnySampleable, context::BATContext)
    algorithm = bat_default_withdebug(bat_marginalmode, Val(:algorithm), measure);
    bat_marginalmode(measure, algorithm, context)
end
