# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct NLSolversFG!{F<:GradientFunction}
    gradfunc::F
end

NLSolversBase.only_fg!(f::GradientFunction) = NLSolversBase.only_fg!(NLSolversFG!(f))

function (gf::NLSolversFG!)(val_f::Real, grad_f::Any, v::Any)
    return gf.gradfunc(!, grad_f, v)
end

function (gf::NLSolversFG!)(val_f::Nothing, grad_f::Any, v::Any)
    gf.gradfunc(!, grad_f, v)
    return Nothing
end


function _bat_findmode_impl_optim(rng::AbstractRNG, target::AnySampleable, algorithm::AbstractModeEstimator)
    transformed_density, trafo = transform_and_unshape(algorithm.trafo, target)

    initalg = apply_trafo_to_init(trafo, algorithm.init)
    x_init = collect(bat_initval(rng, transformed_density, initalg).result)

    f = negative(logdensityof(transformed_density))
    optim_result = _run_optim(f, x_init, algorithm)
    r_optim = Optim.MaximizationWrapper(optim_result)
    transformed_mode = Optim.minimizer(r_optim.res)
    result_mode = inverse(trafo)(transformed_mode)

    dummy_f_x = f(x_init) # ToDo: Avoid recomputation
    trace_trafo = StructArray(;_neg_opt_trace(optim_result, x_init, dummy_f_x) ...)

    (result = result_mode, result_trafo = transformed_mode, trafo = trafo, trace_trafo = trace_trafo, info = r_optim)
end


# Wrapper for type stability of optimize result (why does this work?):
function _optim_optimize(f, x0::AbstractArray, method::Optim.AbstractOptimizer, options = Optim.Options(); kwargs...)
    Optim.optimize(f, x0, method, options; kwargs...)
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
    struct MaxDensityNelderMead <: AbstractModeEstimator

Estimates the mode of a probability density using Nelder-Mead optimization
(currently via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
subject to change).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MaxDensityNelderMead{
    TR<:AbstractDensityTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    trafo::TR = NoDensityTransform()
    init::IA = InitFromTarget()
end
export MaxDensityNelderMead

function _run_optim(f::Function, x_init::AbstractArray{<:Real}, algorithm::MaxDensityNelderMead)
    opts = Optim.Options(store_trace = true, extended_trace=true)
    _optim_optimize(f, x_init, Optim.NelderMead(), opts)
end

bat_findmode_impl(rng::AbstractRNG, target::AnySampleable, algorithm::MaxDensityNelderMead) = _bat_findmode_impl_optim(rng, target, algorithm)


"""
    struct MaxDensityLBFGS <: AbstractModeEstimator

Estimates the mode of a probability density using LBFGS optimization
(currently via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
subject to change).

The gradient of the target density is computed via auto-differentiation.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MaxDensityLBFGS{
    TR<:AbstractDensityTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    trafo::TR = PriorToGaussian()
    init::IA = InitFromTarget()
end
export MaxDensityLBFGS

bat_findmode_impl(rng::AbstractRNG, target::AnySampleable, algorithm::MaxDensityLBFGS) = _bat_findmode_impl_optim(rng, target, algorithm)

function _run_optim(f::Function, x_init::AbstractArray{<:Real}, algorithm::MaxDensityLBFGS)
    fg = valgradof(f)
    opts = Optim.Options(store_trace = true, extended_trace=true)
    _optim_optimize(Optim.only_fg!(fg), x_init, Optim.LBFGS(), opts)
end
