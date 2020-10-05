# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_initval), ::Val{:rng}, ::Any) = bat_rng()
