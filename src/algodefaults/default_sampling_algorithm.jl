# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::BATContext, ::typeof(bat_sample), ::Val{:algorithm}, ::AnyIIDSampleable) = IIDSampling()

bat_default(::BATContext, ::typeof(bat_sample), ::Val{:algorithm}, ::DensitySampleVector) = OrderedResampling()

bat_default(::BATContext, ::typeof(bat_sample), ::Val{:algorithm}, ::AbstractMeasureOrDensity) = MCMCSampling()

function bat_default(context::BATContext, ::typeof(bat_sample), ::Val{:algorithm}, ::DensityMeasure)
    optalg = if get_adselector(context) isa _NoADSelected
        MCMCSampling(mcalg = MetropolisHastings())
    else
        MCMCSampling(mcalg = HamiltonianMC())
    end

    OptimAlg(optalg = optalg)
end
