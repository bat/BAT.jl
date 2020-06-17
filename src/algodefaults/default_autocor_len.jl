# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(
    ::typeof(bat_integrated_autocorr_len),
    ::Val{:algorithm},
    ::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}}
) = GeyerAutocorLen()


bat_default(
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    ::AbstractVectorOfSimilarVectors{<:Real}
) = GeyerAutocorLen()

bat_default(
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    ::DensitySampleVector

) = GeyerAutocorLen()
