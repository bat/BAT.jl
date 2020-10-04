# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_initval), ::Val{:rng}, ::Any) = bat_rng()


function bat_default(::typeof(bat_initval), ::Val{:algorithm}, density::AbstractDensity)
    DT = typeof(density)
    throw(ArgumentError("No default initial value algorithm for density of type $dt available, use ExplicitInit"))
end

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::AbstractPosteriorDensity) = InitFromTarget()

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::Distribution) = InitFromTarget()

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::DistLikeDensity) = InitFromTarget()

bat_default(::typeof(bat_initval), ::Val{:algorithm}, ::SampledDensity) = InitFromSamples()
