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

function _bat_findmode_impl_optim(target::AnySampleable, algorithm::AbstractModeEstimator)
    transformed_density, trafo = transform_and_unshape(algorithm.trafo, target)

    rng = bat_rng()
    initalg = apply_trafo_to_init(trafo, algorithm.init)
    x_init = collect(bat_initval(rng, transformed_density, initalg).result)

    f = negative(logdensityof(transformed_density))
    r_optim = Optim.MaximizationWrapper(_run_optim(f, x_init, algorithm))
    transformed_mode = Optim.minimizer(r_optim.res)
    result_mode = inverse(trafo)(transformed_mode)

    (result = result_mode, result_trafo = transformed_mode, trafo = trafo, info = r_optim)
end


# Wrapper for type stability of optimize result (why does this work?):
function _optim_optimize(f, x0::AbstractArray, method::Optim.AbstractOptimizer, options = Optim.Options(); kwargs...)
    Optim.optimize(f, x0, method, options; kwargs...)
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
    _optim_optimize(f, x_init, Optim.NelderMead())
end

bat_findmode_impl(target::AnySampleable, algorithm::MaxDensityNelderMead) = _bat_findmode_impl_optim(target, algorithm)


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

bat_findmode_impl(target::AnySampleable, algorithm::MaxDensityLBFGS) = _bat_findmode_impl_optim(target, algorithm)

function _run_optim(f::Function, x_init::AbstractArray{<:Real}, algorithm::MaxDensityLBFGS)
    fg = valgradof(f)
    _optim_optimize(Optim.only_fg!(fg), x_init, Optim.LBFGS())
end
