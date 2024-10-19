# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function bat_default(::typeof(bat_sample), ::Val{:algorithm}, target::Any)
    m = convert_for(bat_sample, target)
    if supports_rand(m)
        IIDSampling()
    else
        throw(ArgumentError("Don't know how to sample from objects of type $(nameof(typeof(target)))"))
    end
end

bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::DensitySampleVector) = OrderedResampling()
bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::DensitySampleMeasure) = OrderedResampling()
bat_default(::typeof(bat_sample), ::Val{:algorithm}, ::PosteriorMeasure) = TransformedMCMC()

function bat_default(::typeof(bat_sample), ::Val{:algorithm}, m::EvaluatedMeasure)
    bat_default(bat_sample, Val(:algorithm), m.measure)
end
