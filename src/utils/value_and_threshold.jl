# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ValueAndThreshold{name}

*Experimental feature, not part of stable public API.*

Holds a (target) value, a comparison function and a threshold.

Constructor: `ValueAndThreshold{name}(value, cmp_function, threshold)`

Converts to a `Bool` accoring to `cmp_function(value, threshold)`

Example:

```julia
convert(Bool, ValueAndThreshold{:max_error}(3.4, <, 5.2)) == true
```
"""
struct ValueAndThreshold{name,T,F}
    value::T
    cmpf::F
    threshold::T
end
export ValueAndThreshold

function ValueAndThreshold{name}(value, cmpf::F, threshold) where {name,F}
    conv_value, conv_threshold = promote(value, threshold)
    T = typeof(conv_value)
    ValueAndThreshold{name,T,F}(conv_value, cmpf, conv_threshold)
end

Base.Bool(vt::ValueAndThreshold) = convert(Bool, vt.cmpf(vt.value, vt.threshold))::Bool
Base.convert(::Type{Bool}, vt::ValueAndThreshold) = Bool(vt)
