# This file is a part of BAT.jl, licensed under the MIT License (MIT).


using Compat


@compat abstract type AbstractTargetFunction end


call_target_function(f::Any, params::AbstractVector) = f(params)
call_target_function(f!::AbstractTargetFunction, params::AbstractVector, aux_values::AbstractVector) = f!(aux_values, params)
