# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    Transformed <: AbstractMeasureOrDensity

Abstract type for transformed densities.

In addition to the [`AbstractMeasureOrDensity`](@ref) interface, subtypes must support
the functions `Base.parent` and [`trafoof`](@ref):

```julia
parent(d::SomeTransformedDensity)::AbstractMeasureOrDensity
trafoof(d::SomeTransformedDensity)::Function
```
"""
struct AbstractTransformed end
export AbstractTransformed


"""
    trafoof(d::AbstractTransformed)::AbstractMeasureOrDensity

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
    Transformed

*BAT-internal, not part of stable public API.*
"""
struct Transformed{D<:AbstractMeasureOrDensity,FT<:Function,VC<:TDVolCorr,VS<:AbstractValueShape} <: AbstractMeasureOrDensity
    orig::D
    trafo::FT  # ToDo: store inverse(trafo) instead?
    volcorr::VC
    _varshape::VS
end

function Transformed(orig::AbstractMeasureOrDensity, trafo::Function, volcorr::TDVolCorr)
    vs = trafo(varshape(orig))
    Transformed(orig, trafo, volcorr, vs)
end


@inline function (trafo::DistributionTransform)(density::AbstractMeasureOrDensity; volcorr::Val{vc} = Val(true)) where vc
    if vc
        Transformed(density, trafo, TDLADJCorr())
    else
        Transformed(density, trafo, TDNoCorr())
    end
end


Base.parent(density::Transformed) = density.orig
trafoof(density::Transformed) = density.trafo

@inline DensityInterface.DensityKind(x::Transformed) = DensityKind(x.orig)

ValueShapes.varshape(density::Transformed) = density._varshape

# ToDo: Should not be neccessary, improve default implementation of
# ValueShapes.totalndof(density::AbstractMeasureOrDensity):
ValueShapes.totalndof(density::Transformed) = totalndof(varshape(density))

var_bounds(density::Transformed{<:Any,<:DistributionTransform}) = dist_param_bounds(density.trafo.target_dist)


function DensityInterface.logdensityof(density::Transformed{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inverse(density.trafo)(v)
    logdensityof(parent(density), v_orig)
end

function checked_logdensityof(density::Transformed{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inverse(density.trafo)(v)
    checked_logdensityof(parent(density), v_orig)
end


function _v_orig_and_ladj(density::Transformed, v::Any)
    with_logabsdet_jacobian(inverse(density.trafo), v)
end

# TODO: Would profit from custom pullback:
function _combine_logd_with_ladj(logd_orig::Real, ladj::Real)
    logd_result = logd_orig + ladj
    R = typeof(logd_result)

    if isnan(logd_result) && isneginf(logd_orig) && isposinf(ladj)
        # Zero density wins against infinite volume:
        R(-Inf)
    elseif isfinite(logd_orig) && isneginf(ladj)
        # Maybe  also for isneginf(logd_orig) && isfinite(ladj) ?
        # Return constant -Inf to prevent problems with ForwardDiff:
        #R(-Inf)
        near_neg_inf(R) # Avoids AdvancedHMC warnings
    else
        logd_result
    end
end

function DensityInterface.logdensityof(density::Transformed{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(density, v)
    logd_orig = logdensityof(parent(density), v_orig)
    _combine_logd_with_ladj(logd_orig, ladj)
end

function checked_logdensityof(density::Transformed{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(density, v)
    logd_orig = logdensityof(parent(density), v_orig)
    isnan(logd_orig) && @throw_logged EvalException(logdensityof, density, v, 0)
    _combine_logd_with_ladj(logd_orig, ladj)
end
