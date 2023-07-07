# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::BATContext, ::typeof(bat_convergence), ::Val{:algorithm}, ::AbstractVector{<:DensitySampleVector}) = BrooksGelmanConvergence()
bat_default(::BATContext, ::typeof(bat_convergence), ::Val{:algorithm}, ::DensitySampleVector) = BrooksGelmanConvergence()
