# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using FunctionWrappers: FunctionWrapper


struct GenericProductDensityFunction{T<:Real,P<:Real}
    log_terms::Vector{FunctionWrapper{T,Tuple{P}}}
end

export GenericProductDensityFunction


function density_logval(
    density::GenericProductDensityFunction{T,P},
    params::AbstractVector{<:Real}
) where {T,P}
    sum((log_term(convert(P, T)) for (log_term, p) in (density.log_terms, params)))
end


# For priors, DensityFunction with field sampler_f
# that stores a FunctionWrapper around a sampler()::Sampleable{Multivariate,Continuous}
