# This file is a part of BAT.jl, licensed under the MIT License (MIT).

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
of the stable BAT API.

!!! note

    Do not add add methods to `bat_findmedian`, add methods to
    `bat_findmedian_impl` instead (same arguments).
"""
function bat_findmedian end
export bat_findmedian

function bat_findmedian_impl end


function bat_findmedian(samples::DensitySampleVector)
    r = bat_findmedian_impl(samples::DensitySampleVector)
    result_with_args(r)
end
