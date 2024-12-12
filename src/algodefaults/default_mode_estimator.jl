# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector) = EmpiricalMode()

function bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::MeasureLike)
    optalg = BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    OptimAlg(optalg = optalg)
end

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution) = PredefinedMode()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::BATDistMeasure) = PredefinedMode()

function bat_default(::typeof(bat_findmode), ::Val{:algorithm}, m::EvaluatedMeasure)
    bat_default(bat_findmode, Val(:algorithm), unevaluated(m))
end

bat_default(::typeof(bat_marginalmode), ::Val{:algorithm}, ::DensitySampleVector) = BinnedMode()
