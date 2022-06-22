# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_sample), ::Val{:rng}, ::Any) = bat_rng()


bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::AnyIIDSampleable) = IIDSampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::DensitySampleVector) = OrderedResampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::AbstractMeasureOrDensity) = MCMCSampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::DensityMeasure) = MCMCSampling()
