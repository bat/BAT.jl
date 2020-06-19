# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function bat_findmedian_impl(samples::DensitySampleVector)
    median_params = median(samples)
    (result = median_params,)
end
