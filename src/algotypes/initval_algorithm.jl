# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    abstract type BAT.InitvalAlgorithm

Abstract type for BAT initial/starting value generation algorithms.

Many algorithms in BAT, like MCMC and optimization, need initial/starting
values.
"""
abstract type InitvalAlgorithm end
export InitvalAlgorithm


"""
    bat_initval(
        target::BAT.AnyMeasureOrDensity,
        [algorithm::BAT.InitvalAlgorithm],
        [context::BATContext]
    )::V

    bat_initval(
        target::BAT.AnyMeasureOrDensity,
        n::Integer,
        [algorithm::BAT.InitvalAlgorithm],
        [context::BATContext]
    )::AbstractVector{<:V}

Generate one or `n` random initial/starting value(s) suitable for
`target`.

Assuming the variates of `target` are of type `T`, returns a NamedTuple of
the shape

```julia
(result = X::AbstractVector{T}, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_initval`, add methods like

    ```julia
    bat_initval_impl(target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm, context::BATContext; kwargs...)
    bat_initval_impl(target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm, context::BATContext; kwargs...)
    ```

    to `bat_initval_impl` instead.
"""
function bat_initval end
export bat_initval

function bat_initval_impl end


@inline function bat_initval(target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm, context::BATContext; kwargs...)
    orig_context = deepcopy(context)
    r = bat_initval_impl(target, algorithm, context; kwargs...)
    result_with_args(r, (algorithm = algorithm, context = orig_context), kwargs)
end

@inline function bat_initval(target::AnyMeasureOrDensity; kwargs...)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    context = default_context()
    bat_initval(target, algorithm, context; kwargs...)
end

@inline function bat_initval(target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm; kwargs...)
    context = default_context()
    bat_initval(target, algorithm, context; kwargs...)
end

@inline function bat_initval(target::AnyMeasureOrDensity, context::BATContext; kwargs...)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    bat_initval(target, algorithm, context; kwargs...)
end


@inline function bat_initval(target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm, context::BATContext; kwargs...)
    orig_context = deepcopy(context)
    r = bat_initval_impl(target, n, algorithm, context; kwargs...)
    result_with_args(r, (algorithm = algorithm, context = orig_context), kwargs)
end

@inline function bat_initval(target::AnyMeasureOrDensity, n::Integer; kwargs...)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    context = default_context()
    bat_initval(target, n, algorithm, context; kwargs...)
end

@inline function bat_initval(target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm; kwargs...)
    context = default_context()
    bat_initval(target, n, algorithm, context; kwargs...)
end

@inline function bat_initval(target::AnyMeasureOrDensity, n::Integer, context::BATContext; kwargs...)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    bat_initval(target, n, algorithm, context; kwargs...)
end



function argchoice_msg(::typeof(bat_initval), ::Val{:algorithm}, x::InitvalAlgorithm)
    "Using initval algorithm $x"
end


# Internal for now:
apply_trafo_to_init(trafo::Function, initalg::InitvalAlgorithm) = initalg
