# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: bat_default(::BATContext, ::typeof(bat_integrate), ::Val{:algorithm}, ::AnySampleable) = BridgeSampling()
bat_default(::BATContext, ::typeof(bat_integrate), ::Val{:algorithm}, ::SampledMeasure) = BridgeSampling()
