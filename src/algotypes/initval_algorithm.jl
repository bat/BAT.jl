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
        [rng::AbstractRNG,]
        target::BAT.AnyMeasureOrDensity,
        [algorithm::BAT.InitvalAlgorithm],
    )::V

    bat_initval(
        [rng::AbstractRNG,]
        target::BAT.AnyMeasureOrDensity,
        n::Integer,
        [algorithm::BAT.InitvalAlgorithm],
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
    bat_initval_impl(rng::AbstractRNG, target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm; kwargs...)
    bat_initval_impl(rng::AbstractRNG, target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm; kwargs...)
    ```

    to `bat_initval_impl` instead.
"""
function bat_initval end
export bat_initval

function bat_initval_impl end


@inline function bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm; kwargs...)
    r = bat_initval_impl(rng, target, algorithm; kwargs...)
    result_with_args(r, (rng = rng, algorithm = algorithm), kwargs)
end


@inline function bat_initval(target::AnyMeasureOrDensity; kwargs...)
    rng = bat_default_withinfo(bat_initval, Val(:rng), target)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    bat_initval(rng, target, algorithm; kwargs...)
end


@inline function bat_initval(target::AnyMeasureOrDensity, algorithm::InitvalAlgorithm; kwargs...)
    rng = bat_default_withinfo(bat_initval, Val(:rng), target)
    bat_initval(rng, target, algorithm; kwargs...)
end


@inline function bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity; kwargs...)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    bat_initval(rng, target, algorithm; kwargs...)
end


@inline function bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm; kwargs...)
    r = bat_initval_impl(rng, target, n, algorithm; kwargs...)
    result_with_args(r, (rng = rng, algorithm = algorithm), kwargs)
end


@inline function bat_initval(target::AnyMeasureOrDensity, n::Integer; kwargs...)
    rng = bat_default_withinfo(bat_initval, Val(:rng), target)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    bat_initval(rng, target, n, algorithm; kwargs...)
end


@inline function bat_initval(target::AnyMeasureOrDensity, n::Integer, algorithm::InitvalAlgorithm; kwargs...)
    rng = bat_default_withinfo(bat_initval, Val(:rng), target)
    bat_initval(rng, target, n, algorithm; kwargs...)
end


@inline function bat_initval(rng::AbstractRNG, target::AnyMeasureOrDensity, n::Integer; kwargs...)
    algorithm = bat_default_withinfo(bat_initval, Val(:algorithm), target)
    bat_initval(rng, target, n, algorithm; kwargs...)
end


function argchoice_msg(::typeof(bat_initval), ::Val{:rng}, x::AbstractRNG)
    "Initializing new RNG of type $(typeof(x))"
end

function argchoice_msg(::typeof(bat_initval), ::Val{:algorithm}, x::InitvalAlgorithm)
    "Using initval algorithm $x"
end


# Internal for now:
apply_trafo_to_init(trafo::Function, initalg::InitvalAlgorithm) = initalg
