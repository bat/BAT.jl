# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector) = MaxDensitySearch()

function bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::AbstractMeasureOrDensity)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    OptimAlg(optalg = optalg)
end

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution) = ModeAsDefined()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DistLikeMeasure) = ModeAsDefined()

bat_default(::typeof(bat_marginalmode), ::Val{:algorithm}, ::DensitySampleVector) = BinnedModeEstimator()
