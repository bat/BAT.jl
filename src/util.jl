# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _multi_array_le(a::AbstractArray, b::AbstractArray, c::AbstractArray)
    @inbounds for i in eachindex(a,b,c)
        (a[i] <= b[i] <= c[i]) || return false
    end
    return true
end
