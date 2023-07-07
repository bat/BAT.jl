# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::AnyIIDSampleable) = IIDSampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::DensitySampleVector) = OrderedResampling()

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::BATMeasure) = MCMCSampling()
bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::MeasureBase.DensityMeasure) = MCMCSampling()
