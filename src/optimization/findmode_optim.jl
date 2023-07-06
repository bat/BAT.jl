# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct NLSolversFG!{F,AD} <: Function
    f::F
    ad::AD
end
NLSolversFG!(::Type{FT}, ad::AD) where {FT,AD<:ADSelector}  = NLSolversFG!{Type{FT},AD}(FT, ad)

function (fg!::NLSolversFG!)(::Real, grad_f::Nothing, x::AbstractVector{<:Real})
    y = fg!.f(x)
    return y
end

function (fg!::NLSolversFG!)(::Real, grad_f::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    y, r_grad_f = with_gradient!!(fg!.f, grad_f, x, fg!.ad)
    if !(grad_f === r_grad_f)
        grad_f .= r_grad_f
    end
    return y
end

function (fg!::NLSolversFG!)(::Nothing, grad_f::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    _, r_grad_f = with_gradient!!(fg!.f, grad_f, x, fg!.ad)
    if !(grad_f === r_grad_f)
        grad_f .= r_grad_f
    end
    return Nothing
end


function _bat_findmode_impl_optim(target::AnySampleable, algorithm::AbstractModeEstimator, context::BATContext)
    transformed_density, trafo = transform_and_unshape(algorithm.trafo, target)

    initalg = apply_trafo_to_init(trafo, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)

    f = negative(logdensityof(transformed_density))
    optim_result = _run_optim(f, x_init, algorithm, context)
    r_optim = Optim.MaximizationWrapper(optim_result)
    transformed_mode = Optim.minimizer(r_optim.res)
    result_mode = inverse(trafo)(transformed_mode)

    dummy_f_x = f(x_init) # ToDo: Avoid recomputation
    trace_trafo = StructArray(;_neg_opt_trace(optim_result, x_init, dummy_f_x) ...)

    (result = result_mode, result_trafo = transformed_mode, trafo = trafo, trace_trafo = trace_trafo, info = r_optim)
end


# Wrapper for type stability of optimize result (why does this work?):
function _optim_optimize(f, x0::AbstractArray, method::Optim.AbstractOptimizer, options = Optim.Options())
    Optim.optimize(f, x0, method, options)
end


function _neg_opt_trace(
    @nospecialize(optim_result::Optim.MultivariateOptimizationResults),
    dummy_x::AbstractVector{<:Real}, dummy_f_x::Real
)
    trc = Optim.trace(optim_result)
    tr_len = length(eachindex(trc))
    nd = length(eachindex(dummy_x))

    v = nestedview(similar(dummy_x, nd, tr_len))
    foreach((a,b) -> a[:] = b, v, Optim.x_trace(optim_result))

    logd = similar(dummy_x, typeof(dummy_f_x), tr_len)
    logd[:] = - Optim.f_trace(optim_result)

    if optim_result isa Optim.MultivariateOptimizationResults{<:Optim.ZerothOrderOptimizer}
        (v = v, logd = logd)
    else
        grad_logd = nestedview(similar(dummy_x, nd, tr_len))
        foreach((a,b) -> a[:] = -b.metadata["g(x)"], grad_logd, trc)
        (v = v, logd = logd, grad_logd = grad_logd)
    end
end

function _neg_opt_trace(
    @nospecialize(optim_result::Optim.MultivariateOptimizationResults{<:Optim.NelderMead}),
    dummy_x::AbstractVector{<:Real}, dummy_f_x::Real
)
    trc = Optim.trace(optim_result)
    tr_len = length(eachindex(trc))
    nd = length(eachindex(dummy_x))

    v = nestedview(similar(dummy_x, nd, tr_len))
    foreach((a,b) -> a[:] = b, v, Optim.centroid_trace(optim_result))

    (;v = v)
end


"""
    struct NelderMeadOpt <: AbstractModeEstimator

Estimates the mode of a probability density using Nelder-Mead optimization
(currently via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
subject to change).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct NelderMeadOpt{
    TR<:AbstractTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    trafo::TR = DoNotTransform()
    init::IA = InitFromTarget()
end
export NelderMeadOpt

function _run_optim(f::Function, x_init::AbstractArray{<:Real}, algorithm::NelderMeadOpt, context::BATContext)
    opts = Optim.Options(store_trace = true, extended_trace=true)
    _optim_optimize(f, x_init, Optim.NelderMead(), opts)
end

bat_findmode_impl(target::AnySampleable, algorithm::NelderMeadOpt, context::BATContext) = _bat_findmode_impl_optim(target, algorithm, context)


"""
    struct LBFGSOpt <: AbstractModeEstimator

Estimates the mode of a probability density using LBFGS optimization
(currently via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
subject to change).

The gradient of the target density is computed via auto-differentiation.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct LBFGSOpt{
    TR<:AbstractTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    trafo::TR = PriorToGaussian()
    init::IA = InitFromTarget()
end
export LBFGSOpt

bat_findmode_impl(target::AnySampleable, algorithm::LBFGSOpt, context::BATContext) = _bat_findmode_impl_optim(target, algorithm, context)

function _run_optim(f::Function, x_init::AbstractArray{<:Real}, algorithm::LBFGSOpt, context::BATContext)
    adsel = get_adselector(context)
    if adsel isa _NoADSelected
        throw(ErrorException("LBFGSOpt requires an ADSelector to be specified in the BAT context"))
    end
    fg! = NLSolversFG!(f, adsel)
    opts = Optim.Options(store_trace = true, extended_trace=true)
    _optim_optimize(Optim.only_fg!(fg!), x_init, Optim.LBFGS(), opts)
end
