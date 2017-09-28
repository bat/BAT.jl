# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct GenericProductTargetDensity{T<:Real,P<:Real}
    log_terms::Vector{FunctionWrapper{T,Tuple{P}}}
    single_exec_compat::ExecCapabilities
end

export GenericProductTargetDensity


function target_logval(
    target::GenericProductTargetDensity{T,P},
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
) where {T,P}
    # TODO: Use exec_context and target.exec_capabilities
    sum((log_term(convert(P, T)) for (log_term, p) in (target.log_terms, params)))
end


exec_capabilities(::typeof(target_logval), target::GenericProductTargetDensity, params::AbstractVector{<:Real}) =
    target.single_exec_compat # Change when implementation of target_logval for GenericProductTargetDensity becomes multithreaded.


# ToDo: Add product of target densitys
