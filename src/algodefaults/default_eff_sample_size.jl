# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(
    ::BATContext,
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    samples::AbstractVectorOfSimilarVectors{<:Real},
) = EffSampleSizeFromAC()

bat_default(
    ::BATContext,
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    samples::DensitySampleVector,
) = EffSampleSizeFromAC()
