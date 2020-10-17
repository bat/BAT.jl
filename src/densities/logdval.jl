# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    logvalof(r::NamedTuple{(...,:log,...)})::Real
    logvalof(r::LogDVal)::Real

Extract the log-density value from a result `r`.

Examples:

```julia
logvalof((..., log = log_value, ...)) == log_value
logvalof(LogDVal(log_value)) == log_value
```
"""
function logvalof end
export logvalof

function logvalof(d::Real)
    throw(ArgumentError("Can't get a logarithmic value from $d, unknown if it represents a lin or log value itself."))
end


@inline function logvalof(x::T) where {T<:NamedTuple}
    if hasfield(T, :logval) + hasfield(T, :logd) + hasfield(T, :log) > 1
        throw(ArgumentError("NamedTuples is ambiguous for logvalof contains fields $(join(map(string, filter(name -> name in (:logval, :logd, :log), fieldnames(T))), " and "))"))
    end
    if hasfield(T, :logval)
        x.logval
    elseif hasfield(T, :logd)
        x.logd
    elseif hasfield(T, :log)
        _logvalof_deprecated(x, Val(:log))
    else
        throw(ArgumentError("NamedTuple with fields $(fieldnames(T)) not supported by logvalof, doesn't have a field like :logval"))
    end
end

Base.@noinline function _logvalof_deprecated(x::NamedTuple, ::Val{name}) where name
    Base.depwarn("logvalof support for NamedTuple field $name is deprecated, use NamedTuples with field :logval instead", :logvalof)
    getfield(x, name)
end


"""
    LogDVal{T<:Real}

Represent the logarithm of the value of a statistical density at some point.
`LogDVal` provides means to unambiguously distinguish between linear and
logarithmic values of a density.

Constructor:

    LogDVal(logd::Real)

Use [`logvalof`](@ref) to extract the actual log-density value from
a `LogDVal`:

```julia
    logvalof(LogDVal(d)) == d
```
"""
struct LogDVal{T<:Real}
    logval::T
end

export LogDVal

logvalof(d::LogDVal) = d.logval
