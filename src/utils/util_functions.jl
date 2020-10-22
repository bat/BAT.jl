# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@inline nop_func(x...) = nothing

near_neg_inf(::Type{T}) where T<:Real = T(-1E38) # Still fits into Float32
