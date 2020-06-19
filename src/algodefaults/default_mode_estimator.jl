# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector) = MaxDensitySampleSearch()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::AbstractDensity) = MaxDensityNelderMead()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution) = ModeAsDefined()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DistLikeDensity) = ModeAsDefined()
