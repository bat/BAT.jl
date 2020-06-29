# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    AbstractDensityValue

Represents a value of a density function at some point.

May either be encoded as the linear value (see [`LinDVal`](@ref)) or the log
of the value (see [`LinDVal`](@ref)).

Independent of how the value is stored, it's log value can be retrieved using
[`logvalof`](@ref).
"""
abstract type AbstractDensityValue end
export AbstractDensityValue


"""
    logvalof(d::AbstractDensityValue)::Real

Get the logarithm of density value `d`.

Examples:

```julia
    logvalof(LinDVal(d)) == log(d)
    logvalof(LogDVal(d)) == d
```
"""
function logvalof(d::Real)
    throw(ArgumentError("Can't the a logarithmic value for d, unknown if it represents a lin or log value itself."))
end



"""
    LinDVal{T<:Real} <: AbstractDensityValue

Represent the linear value of a statistical density at some point.
`LinDVal` provides means to unambiguously distinguish between linear and
log result values of density functions.

Constructor:

    LinDVal(d::Real)

Use [`logvalof`](@ref) to extract the logarithm of the density value
from a `LinDVal`:

```julia
    logvalof(LinDVal(d)) == log(d)
```

See also [`LogDVal`](@ref) and [`AbstractDensityValue`](@ref).
"""
struct LinDVal{T<:Real} <: AbstractDensityValue
    value::T
end

export LinDVal

logvalof(d::LinDVal) = log(d.value)



"""
    LogDVal{T<:Real} <: AbstractDensityValue

Represent the logarithm of the value of a statistical density at some point.
`LogDVal` provides means to unambiguously distinguish between linear and
log result values of density functions.

Constructor:

    LogDVal(logd::Real)

Use [`logvalof`](@ref) to extract the actual log-density value from
a `LogDVal`:

```julia
    logvalof(LogDVal(d)) == d
```

See also [`LinDVal`](@ref) and [`AbstractDensityValue`](@ref).
"""
struct LogDVal{T<:Real} <: AbstractDensityValue
    value::T
end

export LogDVal

logvalof(d::LogDVal) = d.value
