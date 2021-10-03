# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    TransformedDensity <: AbstractDensity

Abstract type for transformed densities.

In addition to the [`AbstractDensity`](@ref) interface, subtypes must support
the functions `Base.parent` and [`trafoof`](@ref):

```julia
parent(d::SomeTransformedDensity)::AbstractDensity
trafoof(d::SomeTransformedDensity)::AbstractVariateTransform
```
"""
struct AbstractTransformedDensity end
export AbstractTransformedDensity


"""
    trafoof(d::AbstractTransformedDensity)::AbstractDensity

Get the transform from `parent(d)` to `d`, so that

```julia
trafoof(d)(parent(d)) == d
```
"""
function trafoof end
export trafoof


abstract type TDVolCorr end
struct TDNoCorr <: TDVolCorr end
struct TDLADJCorr <: TDVolCorr end


"""
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

ValueShapes.varshape(density::TransformedDensity) = valshape(density.trafo)

# ToDo: Should not be neccessary, improve default implementation of
# ValueShapes.totalndof(density::AbstractDensity):
ValueShapes.totalndof(density::TransformedDensity) = totalndof(varshape(density))

var_bounds(density::TransformedDensity{<:Any,<:DistributionTransform}) = dist_param_bounds(density.trafo.target_dist)


function DensityInterface.logdensityof(density::TransformedDensity{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inv(density.trafo)(v)
    logdensityof(parent(density), v_orig)
end

function checked_logdensityof(density::TransformedDensity{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inv(density.trafo)(v)
    checked_logdensityof(parent(density), v_orig)
end


function _v_orig_and_ladj(density::TransformedDensity, v::Any)
    r = inv(density.trafo)(v, 0)
    r.v, r.ladj
end

# TODO: Would profit from custom pullback:
function _combine_logd_with_ladj(logd_orig::Real, ladj::Real)
    logd_result = logd_orig + ladj
    R = typeof(logd_result)

    if isnan(logd_result) && logd_orig == -Inf && ladj == +Inf
        # Zero density wins against infinite volume:
        R(-Inf)
    elseif isfinite(logd_orig) && (ladj == -Inf)
        # Maybe  also for (logd_orig == -Inf) && isfinite(ladj) ?
        # Return constant -Inf to prevent problems with ForwardDiff:
        #R(-Inf)
        near_neg_inf(R) # Avoids AdvancedHMC warnings
    else
        logd_result
    end
end

function DensityInterface.logdensityof(density::TransformedDensity{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(density, v)
    logd_orig = logdensityof(parent(density), v_orig)
    _combine_logd_with_ladj(logd_orig, ladj)
end

function checked_logdensityof(density::TransformedDensity{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(density, v)
    logd_orig = checked_logdensityof(parent(density), v_orig)
    _combine_logd_with_ladj(logd_orig, ladj)
end
