# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    bat_findmedian(
        samples::DensitySampleVector
    )::DensitySampleVector

The function computes the median of marginalized `samples`.

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, ...)
```
"""
function bat_findmedian end
export bat_findmedian
