# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_integrate), ::Val{:algorithm}, ::AnyPosterior) = AHMIntegration()

