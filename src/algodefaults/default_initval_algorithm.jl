# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::BATContext, ::typeof(bat_initval), ::Val{:algorithm}, ::AnyMeasureOrDensity) = InitFromTarget()
