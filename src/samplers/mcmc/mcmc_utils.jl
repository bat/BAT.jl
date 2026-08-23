# This file is a part of BAT.jl, licensed under the MIT License (MIT).



# The pulled-back log-density and gradient of the target under an affine
# transform x = A z + b, via the analytic chain rule
# log pi_z(z) = log pi_x(A z + b) + logabsdet(A) and grad_z = A' grad_x.
# AD only ever differentiates the fixed-space target, so the AD
# preparation stays valid across geometry changes, and operator-valued
# affine transforms are supported without AD seeing the operator:
struct _AffinePullbackValGrad{FG,FT<:MulAdd,T<:Real} <: Function
    fg_x::FG
    f_transform::FT
    ladj::T
end

function (fg::_AffinePullbackValGrad)(z::AbstractVector{<:Real})
    x = fg.f_transform(z)
    logd_x, grad_x = fg.fg_x(x)
    return logd_x + fg.ladj, fg.f_transform.A' * grad_x
end

function _target_logdgrad_func(target::BATMeasure, f_transform::MulAdd, context::BATContext, proposal_alg, x_dummy::AbstractVector{<:Real})
    adsel = get_valid_adselector(context, proposal_alg)
    fg_x = valgrad_func(checked_logdensityof(target), adsel, x_dummy)
    return _AffinePullbackValGrad(fg_x, f_transform, first(logabsdet(f_transform.A)))
end

function _target_logdgrad_func(target::BATMeasure, ::typeof(identity), context::BATContext, proposal_alg, x_dummy::AbstractVector{<:Real})
    adsel = get_valid_adselector(context, proposal_alg)
    return valgrad_func(checked_logdensityof(target), adsel, x_dummy)
end

# Generic (possibly nonlinear) transforms differentiate through the full
# pullback:
function _target_logdgrad_func(target::BATMeasure, f_transform::Function, context::BATContext, proposal_alg, x_dummy::AbstractVector{<:Real})
    adsel = get_valid_adselector(context, proposal_alg)
    f = checked_logdensityof(MeasureBase.pullback(f_transform, target))
    return valgrad_func(f, adsel)
end

# Updated valgrad function after a transform change: the fixed-space AD
# preparation stays valid across affine geometry changes, only the affine
# wrapper is rebuilt:
function _updated_logdgrad_func(fg_old, target::BATMeasure, f_new::Function, context::BATContext, proposal_alg, x_dummy::AbstractVector{<:Real})
    if fg_old isa _AffinePullbackValGrad && f_new isa MulAdd
        return _AffinePullbackValGrad(fg_old.fg_x, f_new, first(logabsdet(f_new.A)))
    else
        return _target_logdgrad_func(target, f_new, context, proposal_alg, x_dummy)
    end
end
