# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type BAT.AbstractMedianEstimator

Abstract type for BAT optimization algorithms.

A typical application for optimization in BAT is mode estimation
(see [`bat_findmode`](@ref)),
"""
abstract type AbstractMedianEstimator end


"""
    bat_findmedian(
        samples::DensitySampleVector
    )

The function computes the median of marginalized `samples`.

Returns a NamedTuple of the shape

```julia
(result = v, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_findmedian`, add methods to
    `bat_findmedian_impl` instead.
"""
function bat_findmedian end
export bat_findmedian

function bat_findmedian_impl end

function bat_findmedian(samples::DensitySampleVector, algorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_findmedian_impl(samples, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_findmedian(samples::DensitySampleVector) = bat_findmedian(samples, get_batcontext())

bat_findmedian(samples::DensitySampleVector, algorithm) = bat_findmedian(samples, algorithm, get_batcontext())

function bat_findmedian(samples::DensitySampleVector, context::BATContext)
    algorithm = bat_default_withdebug(bat_findmedian, Val(:algorithm), samples)
    bat_findmedian(samples, algorithm, context)
end

