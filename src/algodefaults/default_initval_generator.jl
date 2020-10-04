# This file is a part of BAT.jl, licensed under the MIT License (MIT).

bat_default(::typeof(bat_initval), ::Val{:rng}, ::Any) = bat_rng()


bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::DensitySampleVector) = MaxDensitySampleSearch()

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::AbstractDensity) = MaxDensityNelderMead()

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::Distribution) = ModeAsDefined()

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::DistLikeDensity) = ModeAsDefined()

