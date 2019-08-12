# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add `density_logvalgrad!` to support HMC, etc.


@doc """
    AbstractDensity

The following functions must be implemented for subtypes:

* `BAT.nparams`
* `BAT.density_logval`
"""
abstract type AbstractDensity end
export AbstractDensity


Random.rand(rng::AbstractRNG, density::AbstractDensity, T::Type{<:AbstractFloat}) =
    rand!(rng, density, Vector{T}(undef, nparams(density)))

Random.rand(rng::AbstractRNG, density::AbstractDensity, T::Type{<:AbstractFloat}, n::Integer) =
    rand!(rng, density, VectorOfSimilarVectors(Array{T}(undef, nparams(density), n)))

Random.rand!(rng::AbstractRNG, density::AbstractDensity, x::AbstractVector{<:Real}) =
    rand!(rng, sampler(density), x)

Random.rand!(rng::AbstractRNG, density::AbstractDensity, x::VectorOfSimilarVectors{<:Real}) =
    (rand!(rng, sampler(density), flatview(x)); x)



@doc """
    param_bounds(density::AbstractDensity)::AbstractParamBounds

Get the parameter bounds of `density`. See `density_logval` for the
implications and handling of the bounds.

Use

   new_density = density[bounds::ParamVolumeBounds]

to create a new density function with additional bounds.
"""
function param_bounds(density::AbstractDensity)
    NoParamBounds(nparams(density))
end
export param_bounds



@doc """
    density_logval(
        density::AbstractDensity,
        params::AbstractVector{<:Real}
    )

Compute log of value of a multi-variate density function at the given
parameter values.

Input:

* `density`: density function
* `params`: parameter values

Note: If `density_logval` is called with out-of-bounds parameters (see
`param_bounds`), the behaviour is undefined. The result for parameters that
are not within bounds is *implicitly* `-Inf`, but it is the caller's
responsibility to handle these cases.
"""
function density_logval end
export density_logval


@doc """
    eval_density_logval!(
        T::Type{<:Real},
        density::AbstractDensity,
        params::AbstractVector{<:Real},
    )

Apply bounds and then evaluate density and check return value.

May modify `params` to force them into bounds.

Guarantees that for out-of-bounds parameters:

* `density_logval` is not called
* log value of density is set to (resp. returned as) `-Inf`
"""
function eval_density_logval! end

function eval_density_logval!(
    T::Type{<:Real},
    density::AbstractDensity,
    params::AbstractVector{<:Real},
)
    (length(eachindex(params)) != nparams(density)) && throw(ArgumentError("Invalid length of parameter vector"))

    bounds = param_bounds(density)
    apply_bounds!(params, bounds)
    if !isoob(params)
        @assert params in bounds  # TODO: Remove later on for increased performance, should never trigger
        r = density_logval(density, params)
        isnan(r) && throw(ErrorException("Return value of density_logval must not be NaN"))  
        T(r)
    else
        T(-Inf)
    end
end




@doc """
    GenericDensity{F} <: AbstractDensity

Constructors:

    GenericDensity(log_f, nparams::Int)

Turns the logarithmic density function `log_f` into a
BAT-compatible `AbstractDensity`. `log_f` must support

    `log_f(params::AbstractVector{<:Real})::Real`

with `length(params) == nparams`.

It must be safe to execute `log_f` in parallel on multiple threads and
processes.
"""
struct GenericDensity{F} <: AbstractDensity
    log_f::F
    nparams::Int
end

export GenericDensity

Base.parent(density::GenericDensity) = density.log_f

nparams(density::GenericDensity) = density.nparams

function density_logval(
    density::GenericDensity,
    params::AbstractVector{<:Real}
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    density.log_f(params)
end



#=

# TODO:

mutable struct TransformedDensity{...}{
    SO<:AbstractDensity,
    SN<:AbstractDensity
} <: AbstractDensity
   before::SO
   after::SN
   # ... transformation, Jacobi matrix of transformation, etc.
end

export TransformedDensity


function rand!(rng::AbstractRNG, density::TransformedDensity, x::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}}) =
    initial_params!(rng, algorithm, density.before, x)
    ... apply transformation to x ...
    x
end

...

=#
