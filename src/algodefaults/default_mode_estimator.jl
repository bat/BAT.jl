# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector, context) = MaxDensitySearch()

function bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::AnyMeasureLike, context)
    optalg = if get_adselector(context) isa _NoADSelected
        BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    else
        BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG))
    end

    OptimAlg(optalg = optalg)
end

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution, context) = ModeAsDefined()

bat_default(::typeof(bat_findmode), ::Val{:algorithm}, ::DistLikeMeasure, context) = ModeAsDefined()
