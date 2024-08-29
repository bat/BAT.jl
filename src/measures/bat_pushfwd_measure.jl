# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BATPushFwdMeasure

*BAT-internal, not part of stable public API.*
"""
struct BATPushFwdMeasure{F,I,M<:BATMeasure,VC<:PushFwdStyle} <: BATMeasure
    f :: F
    finv :: I
    origin :: M
    volcorr :: VC
end

const _NonBijectiveBATPusfwdMeasure{M<:BATMeasure,VC<:PushFwdStyle} = Union{
    BATPushFwdMeasure{<:Any,<:NoInverse,M,VC},
    BATPushFwdMeasure{<:NoInverse,<:Any,M,VC},
    BATPushFwdMeasure{<:NoInverse,<:NoInverse,M,VC}
}


function BATPushFwdMeasure(f, origin::BATMeasure, volcorr::PushFwdStyle)
    BATPushFwdMeasure(f, inverse(f), origin, volcorr)
end

BATMeasure(m::PushforwardMeasure) = BATPushFwdMeasure(m.f, m.finv, batmeasure(m.origin), m.volcorr)

MeasureBase.gettransform(m::BATPushFwdMeasure) = m.trafo

MeasureBase.transport_origin(m::BATMeasure) = m.orig
MeasureBase.from_origin(m::BATMeasure, x) = m.f(x)
MeasureBase.to_origin(m::BATMeasure, y) = m.finv(y)

MeasureBase.getdof(m::BATPushFwdMeasure) = getdof(m.orig)
MeasureBase.getdof(m::_NonBijectiveBATPusfwdMeasure) = MeasureBase.NoDOF{typeof(m)}()

MeasureBase.insupport(m::BATPushFwdMeasure, x) = insupport(transport_origin(m), to_origin(m, x))

MeasureBase.massof(m::BATPushFwdMeasure) = massof(transport_origin(m))

MeasureBase.rootmeasure(m::BATPushFwdMeasure{F,I,M,ChangeRootMeasure}) where {F,I,M} = pushfwd(m.f, rootmeasure(m.origin))
MeasureBase.rootmeasure(m::BATPushFwdMeasure{F,I,M,KeepRootMeasure}) where {F,I,M} = rootmeasure(m.origin)


MeasureBase.pushfwd(f, m::BATMeasure) = _bat_pushfwd(f, m, KeepRootMeasure())
MeasureBase.pushfwd(f, m::BATMeasure, volcorr::KeepRootMeasure) = _bat_pushfwd(f, m, volcorr)
MeasureBase.pushfwd(f, m::BATMeasure, volcorr::ChangeRootMeasure) = _bat_pushfwd(f, m, volcorr)

_bat_pushfwd(f, m::BATMeasure, volcorr::PushFwdStyle) = BATPushFwdMeasure(f, m, volcorr)

function _bat_pushfwd(f, m::BATPushFwdMeasure{F,I,M,VC}, volcorr::VC) where {F,I,M,VC}
    BATPushFwdMeasure(fcomp(f, m.f), fcomp(m.finv, inverse(f)), m, volcorr)
end

_bat_pushfwd(::typeof(identity), m::BATMeasure, ::KeepRootMeasure) = m
_bat_pushfwd(::typeof(identity), m::BATMeasure, ::ChangeRootMeasure) = m


MeasureBase.pullback(f, m::BATMeasure) = _bat_pulbck(f, m, KeepRootMeasure())
MeasureBase.pullback(f, m::BATMeasure, volcorr::KeepRootMeasure) = _bat_pulbck(f, m, volcorr)
MeasureBase.pullback(f, m::BATMeasure, volcorr::ChangeRootMeasure) = _bat_pulbck(f, m, volcorr)

_bat_pulbck(f, m::BATMeasure, volcorr::PushFwdStyle) = MeasureBase.pushfwd(inverse(f), m, volcorr)


# ToDo: remove
function (f::DistributionTransform)(m::AbstractMeasure; volcorr::Val{vc} = Val(true)) where vc
    throw(ErrorException("`(f::BAT.DistributionTransform)(measure)` is no longer supported, use `MeasureBase.pushfwd(f, measure)` instead."))
end


#!!!!!!!!! Use return type of trafo with testvalue, if no shape change return varshape(m.orig) directly
ValueShapes.varshape(m::BATPushFwdMeasure) = trafo(varshape(m.orig))

ValueShapes.varshape(m::BATPushFwdMeasure{<:DistributionTransform}) = varshape(m.f.target_dist)


measure_support(m::BATPushFwdMeasure{<:DistributionTransform}) = dist_support(m.f.target_dist)


function DensityInterface.logdensityof(@nospecialize(m::_NonBijectiveBATPusfwdMeasure{M,<:ChangeRootMeasure}), @nospecialize(v::Any)) where M
    throw(ArgumentError("Can't calculate densities for non-bijective pushforward measure $(nameof(M))"))
end

function DensityInterface.logdensityof(m::BATPushFwdMeasure{F,I,M,ChangeRootMeasure}, v::Any) where {F,I,M}
    v_orig = m.finv(v)
    logdensityof(m.origin, v_orig)
end

function checked_logdensityof(m::BATPushFwdMeasure{F,I,M,ChangeRootMeasure}, v::Any) where {F,I,M}
    v_orig = m.finv(v)
    checked_logdensityof(m.origin, v_orig)
end


function _v_orig_and_ladj(m::BATPushFwdMeasure, v::Any)
    with_logabsdet_jacobian(m.finv, v)
end

# TODO: Would profit from custom pullback:
function _combine_logd_with_ladj(logd_orig::Real, ladj::Real)
    logd_result = logd_orig + ladj
    R = typeof(logd_result)

    if isnan(logd_result) && isneginf(logd_orig) && isposinf(ladj)
        # Zero m wins against infinite volume:
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


function DensityInterface.logdensityof(@nospecialize(m::_NonBijectiveBATPusfwdMeasure{M,<:KeepRootMeasure}), @nospecialize(v::Any)) where M
    throw(ArgumentError("Can't calculate densities for non-bijective pushforward measure $(nameof(M))"))
end

function DensityInterface.logdensityof(m::BATPushFwdMeasure{F,I,M,KeepRootMeasure}, v::Any) where {F,I,M}
    v_orig, ladj = _v_orig_and_ladj(m, v)
    logd_orig = logdensityof(m.origin, v_orig)
    _combine_logd_with_ladj(logd_orig, ladj)
end

function checked_logdensityof(m::BATPushFwdMeasure{F,I,M,KeepRootMeasure}, v::Any) where {F,I,M}
    v_orig, ladj = _v_orig_and_ladj(m, v)
    logd_orig = logdensityof(m.origin, v_orig)
    isnan(logd_orig) && @throw_logged EvalException(logdensityof, m, v, 0)
    _combine_logd_with_ladj(logd_orig, ladj)
end


Random.rand(rng::AbstractRNG, ::Type{T}, m::BATPushFwdMeasure) where {T<:Real} = m.f(rand(rng, T, m.origin))

Random.rand(rng::AbstractRNG, m::BATPushFwdMeasure) = m.f(rand(rng, m.origin))

supports_rand(m::BATPushFwdMeasure) = supports_rand(m.origin)


_dist_with_pushfwd(m::BATPushFwdMeasure) = _dist_with_pushfwd_impl(m.origin, m.f)
_dist_with_pullback(m::BATPushFwdMeasure) = _dist_with_pullback_impl(m.origin, m.finv)
