# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using FunctionWrappers: FunctionWrapper


struct GenericProductDensityFunction{T<:Real,P<:Real}
    log_terms::Vector{FunctionWrapper{T,Tuple{P}}}
    single_exec_compat::ExecCapabilities
end

export GenericProductDensityFunction


function unsafe_density_logval(
    density::GenericProductDensityFunction{T,P},
    params::AbstractVector{<:Real},
    exec_context::ExecContext
) where {T,P}
    # TODO: Use exec_context and density.exec_capabilities
    sum((log_term(convert(P, T)) for (log_term, p) in (density.log_terms, params)))
end


exec_capabilities(::typeof(unsafe_density_logval), density::GenericProductDensityFunction, params::AbstractVector{<:Real}) =
    density.single_exec_compat # Change when implementation of density_logval for GenericProductDensityFunction becomes multithreaded.


# For priors, DensityFunction with field sampler_f
# that stores a FunctionWrapper around a sampler()::Sampleable{Multivariate,Continuous}
