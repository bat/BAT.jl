# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    BAT.InitvalGenerator

Abstract type for BAT initial/starting value generation algorithms.

Many algorithms in BAT, like MCMC and optimization need initial starting
values.
"""
abstract type InitvalGenerator end


"""
    bat_initval(
        [rng::AbstractRNG,]
        target::BAT.AnySampleable,
        [algorithm::BAT.InitvalGenerator],
    )::V

    bat_initval(
        [rng::AbstractRNG,]
        target::BAT.AnySampleable, n::Integer,
        [algorithm::BAT.InitvalGenerator],
    )::AbstractVector{<:V}

Generate one or `n` random initial/starting value(s) suitable for
`target`.

Assuming the variates of `target` are of type `T`, returns a NamedTuple of
the shape

```julia
(result = X::AbstractVector{T}, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.

!!! note

    Do not add add methods to `bat_initval`, add methods to
    `bat_initval_impl` instead (same arguments).
"""
function bat_initval end
export bat_initval

function bat_initval_impl end


function bat_initval(
    target::AnySampleable,
    algorithm = bat_default_withdebug(bat_initval, Val(:algorithm), target);
    kwargs...
)
    r = bat_initval_impl(target, algorithm; kwargs...)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_initval), ::Val{:algorithm}, x::InitvalGenerator)
    "Using initial value generator $x"
end
