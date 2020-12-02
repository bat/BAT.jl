# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    TransformedDensity <: AbstractDensity

Abstract super-type for transformed densities.

In addition to the [`AbstractDensity`](@ref) interface, subtypes must support
the functions `Base.parent` and [`trafoof`](@ref):

```julia
parent(d::SomeTransformedDensity)::AbstractDensity
trafoof(d::SomeTransformedDensity)::AbstractVariateTransform
```
"""
struct AbstractTransformedDensity end
export AbstractTransformedDensity


@doc doc"""
    trafoof(d::AbstractTransformedDensity)::AbstractDensity

Get the transform from `parent(d)` to d, so that

```julia
trafoof(parent(d)) == d
```
"""
function trafoof end
export trafoof


abstract type TDVolCorr end
struct TDNoCorr <: TDVolCorr end
struct TDLADJCorr <: TDVolCorr end


@doc doc"""
    TransformedDensity

*BAT-internal, not part of stable public API.*
"""
struct TransformedDensity{D<:AbstractDensity,FT<:VariateTransform,VC<:TDVolCorr} <: AbstractDensity
    orig::D
    trafo::FT
    volcorr::VC

    # ToDo: Check varshape(orig) == varshape(trafo) in ctor
end


@inline function (trafo::VariateTransform)(density::AbstractDensity; volcorr::Val{vc} = Val(true)) where vc
    if vc
        TransformedDensity(density, trafo, TDLADJCorr())
    else
        TransformedDensity(density, trafo, TDNoCorr())
    end
end


Base.parent(density::TransformedDensity) = density.orig
trafoof(density::TransformedDensity) = density.trafo

# var_bounds(density::TransformedDensity) = ...

ValueShapes.varshape(density::TransformedDensity{<:AbstractDensity,<:DistributionTransform}) = varshape(density.trafo.target_dist)

# ToDo: Should not be neccessary, improve default implementation of
# ValueShapes.totalndof(density::AbstractDensity):
ValueShapes.totalndof(density::TransformedDensity) = totalndof(varshape(density))


var_bounds(density::TransformedDensity) = var_bounds(density.trafo)

function var_bounds(trafo::VariateTransform)
    n = totalndof(varshape(trafo))
    HyperRectBounds(fill(_default_PT(-Inf), n), fill(_default_PT(+Inf), n), hard_bounds)
end

function var_bounds(trafo::VariateTransform{<:VariateForm,<:UnitSpace})
    n = totalndof(varshape(trafo))
    HyperRectBounds(fill(_default_PT(0), n), fill(_default_PT(1), n), hard_bounds)
end


function eval_logval(
    density::TransformedDensity{D,FT,<:TDNoCorr},
    v::Any,
    T::Type{<:Real} = density_logval_type(v);
    use_bounds::Bool = true,
    strict::Bool = false
) where {D,FT,}
    v_shaped = reshape_variate(varshape(density), v)
    v_orig = inv(density.trafo)(v_shaped)
    eval_logval(parent(density), v_orig, T, use_bounds = use_bounds, strict = strict)
end


function eval_logval(
    density::TransformedDensity{D,FT,<:TDLADJCorr},
    v::Any,
    T::Type{<:Real} = density_logval_type(v);
    use_bounds::Bool = true,
    strict::Bool = false
) where {D,FT,}
    v_shaped = reshape_variate(varshape(density), v)
    r = inv(density.trafo)(v_shaped, 0)
    v_orig = r.v
    ldaj = r.ladj
    logd_orig = eval_logval(parent(density), v_orig, T, use_bounds = use_bounds, strict = strict)

    logd_result = logd_orig + ldaj
    R = typeof(logd_result)

    if isnan(logd_result) && logd_orig == -Inf && ldaj == +Inf
        # Zero density wins against infinite volume:
        R(-Inf)
    elseif isfinite(logd_orig) && (ldaj == -Inf)
        # Maybe  also for (logd_orig == -Inf) && isfinite(ldaj) ?
        # Return constant -Inf to prevent problems with ForwardDiff:
        #R(-Inf)
        near_neg_inf(R) # Avoids AdvancedHMC warnings
    else
        logd_result
    end
end


function eval_logval_unchecked(density::TransformedDensity, v::Any)
    eval_logval(density, v, use_bounds = false, strict = false)
end
