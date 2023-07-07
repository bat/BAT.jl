# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::BATContext, ::typeof(bat_findmode), ::Val{:algorithm}, ::DensitySampleVector) = MaxDensitySearch()

function bat_default(context::BATContext, ::typeof(bat_findmode), ::Val{:algorithm}, ::AbstractMeasureOrDensity)
    optalg = if get_adselector(context) isa _NoADSelected
        BAT.ext_default(pkgext(Val(:Optim)), Val(:NELDERMEAD_ALG))
    else
        BAT.ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG))
    end

    OptimAlg(optalg = optalg)
end

bat_default(::BATContext, ::typeof(bat_findmode), ::Val{:algorithm}, ::Distribution) = ModeAsDefined()

bat_default(::BATContext, ::typeof(bat_findmode), ::Val{:algorithm}, ::DistLikeMeasure) = ModeAsDefined()

bat_default(::BATContext, ::typeof(bat_marginalmode), ::Val{:algorithm}, ::DensitySampleVector) = BinnedModeEstimator()
