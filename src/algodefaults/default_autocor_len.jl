# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(
    ::BATContext,
    ::typeof(bat_integrated_autocorr_len),
    ::Val{:algorithm},
    ::Union{AbstractVector{<:Real},AbstractVectorOfSimilarVectors{<:Real}},
) = GeyerAutocorLen()
