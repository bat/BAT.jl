# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector) = MaxDensitySearch()

function bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::MeasureLike)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    TransformedMaxDensity(optalg = OptimAlg(optalg = optalg))
end

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution) = ModeAsDefined()

bat_default(::typeof(bat_bgml), ::Val{:algorithm}, likelihood, prior) = bat_default(bat_findmode, Val(:algorithm), lbqintegral(likelihood, prior))

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::BATDistMeasure) = ModeAsDefined()

function bat_default(::typeof(bat_findmode), ::Val{:algorithm}, m::EvaluatedMeasure)
    bat_default(bat_findmode, Val(:algorithm), unevaluated(m))
end

bat_default(::typeof(bat_marginalmode), ::Val{:algorithm}, ::DensitySampleVector) = BinnedModeEstimator()
