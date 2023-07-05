# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector) = MaxDensitySearch()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::AbstractMeasureOrDensity) = NelderMeadOpt()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution) = ModeAsDefined()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DistLikeMeasure) = ModeAsDefined()
