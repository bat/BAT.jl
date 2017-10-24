# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct BoundedDensity{
    F<:AbstractDensityFunction,
    B<:AbstractParamBounds
} #<: AbstractBoundedDensity
    density::F
    bounds::B
end

export BoundedDensity

Base.parent(density::BoundedDensity) = density.density

param_bounds(density::BoundedDensity) = density.bounds
nparams(density::BoundedDensity) = nparams(density.bounds)



function density_logval(
    density::BoundedDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext(),
    check_bounds = true
)
    l = density_logval(parent(density), params, exec_context)
    if !check_bounds || params in param_bounds(density)
        l
    else
        typeof(l)(-Inf)
    end
end

exec_capabilities(::typeof(density_logval), density::BoundedDensity, params::AbstractVector{<:Real}) =
    exec_capabilities(density_logval, parent(density), params)


function density_logval!(
    r::AbstractArray{<:Real},
    density::AbstractDensityFunction,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext(),
    check_bounds = true
)
    ret = density_logval!(r, parent(density), params, exec_context)

    if (check_bounds)
        # TODO: Set entries of r to -Inf if respective param vector not in bounds.
        error("Bounds check not implemented yet")
    end

    ret
end

exec_capabilities(::typeof(density_logval!), density::BoundedDensity, params::AbstractMatrix{<:Real}) =
    exec_capabilities(density_logval!, parent(density), params::AbstractMatrix{<:Real})
