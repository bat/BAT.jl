# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type TDVolCorr end
struct TDNoCorr <: TDVolCorr end
struct TDLADJCorr <: TDVolCorr end


"""
    Transformed

*BAT-internal, not part of stable public API.*
"""
struct Transformed{D<:BATMeasure,FT<:Function,VC<:TDVolCorr,VS<:AbstractValueShape} <: BATMeasure
    orig::D
    trafo::FT  # ToDo: store inverse(trafo) instead?
    volcorr::VC
    _varshape::VS
end

function Transformed(orig::BATMeasure, trafo::Function, volcorr::TDVolCorr)
    vs = trafo(varshape(orig))
    Transformed(orig, trafo, volcorr, vs)
end


@inline function (trafo::DistributionTransform)(origin::AnyMeasureLike; volcorr::Val{vc} = Val(true)) where vc
    measure = BATMeasure(origin)
    if vc
        Transformed(measure, trafo, TDLADJCorr())
    else
        Transformed(measure, trafo, TDNoCorr())
    end
end


Base.parent(measure::Transformed) = measure.orig

@inline DensityInterface.DensityKind(x::Transformed) = DensityKind(x.orig)

ValueShapes.varshape(measure::Transformed) = measure._varshape

# ToDo: Should not be neccessary, improve default implementation of
# ValueShapes.totalndof(measure::BATMeasure):
ValueShapes.totalndof(measure::Transformed) = totalndof(varshape(measure))


function DensityInterface.logdensityof(measure::Transformed{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inverse(measure.trafo)(v)
    logdensityof(parent(measure), v_orig)
end

function checked_logdensityof(measure::Transformed{D,FT,TDNoCorr}, v::Any) where {D,FT}
    v_orig = inverse(measure.trafo)(v)
    checked_logdensityof(parent(measure), v_orig)
end


function _v_orig_and_ladj(measure::Transformed, v::Any)
    with_logabsdet_jacobian(inverse(measure.trafo), v)
end

# TODO: Would profit from custom pullback:
function _combine_logd_with_ladj(logd_orig::Real, ladj::Real)
    logd_result = logd_orig + ladj
    R = typeof(logd_result)

    if isnan(logd_result) && isneginf(logd_orig) && isposinf(ladj)
        # Zero measure wins against infinite volume:
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

function BAT.logdensityof_batmeasure(measure::Transformed{D,FT,TDLADJCorr}, v::Any) where {D,FT,}
    v_orig, ladj = _v_orig_and_ladj(measure, v)
    logd_orig = logdensityof(parent(measure), v_orig)
    isnan(logd_orig) && @throw_logged EvalException(logdensityof, measure, v, 0)
    _combine_logd_with_ladj(logd_orig, ladj)
end
