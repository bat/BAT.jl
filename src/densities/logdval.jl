# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    LinDVal{T<:Real}

Represent the linear value of a statistical density at some point.
`LinDVal` provides means to unambiguously distinguish between linear and
log result values of density functions.

Constructor:

    LinDVal(d::Real)

Use [`BAT.density_logval`](@ref) to extract the logarithm of the density value
from a `LinDVal`:

```julia
    BAT.density_logval(LinDVal(logd)) == logd
```

See also [`LogDVal`](@ref), [`AbstractDensity`](@ref) and
[`BAT.GenericDensity`](@ref).
"""
struct LinDVal{T<:Real}
    value::T
end

export LinDVal


"""
    BAT.density_logval(x::LinDVal) = log(x.value)

Get the logarithm 

Examples:

```julia
    BAT.density_logval(LinDVal(x)) == log(x)
    BAT.density_logval(LogDVal(x)) == x
```
    
"""
density_logval(x::LinDVal) = log(x.value)



"""
    LogDVal{T<:Real}

Represent the logarithm of the value of a statistical density at some point.
`LogDVal` provides means to unambiguously distinguish between linear and
log result values of density functions.

Constructor:

    LogDVal(logd::Real)

Use [`BAT.density_logval`](@ref) to extract the actual log-density value from
a `LogDVal`:

```julia
    BAT.density_logval(LogDVal(logd)) == logd
```

See also [`LogDVal`](@ref), [`AbstractDensity`](@ref) and
[`BAT.GenericDensity`](@ref).
"""
struct LogDVal{T<:Real}
    value::T
end

export LogDVal


density_logval(x::LogDVal) = x.value



# ToDo: Add LogDGrad
