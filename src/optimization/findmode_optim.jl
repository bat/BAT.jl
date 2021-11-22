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
    density_notrafo = convert(AbstractDensity, target)
    trafoalg = bat_default(bat_transform, Val(:algorithm), algorithm.trafo, density_notrafo)
    shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo, trafoalg)
    shape = varshape(shaped_density)
    density = unshaped(shaped_density)

    rng = bat_determ_rng()
    x_init = collect(unshaped(bat_initval(rng, density, apply_trafo_to_init(trafo, algorithm.init)).result))

    f = negative(logdensityof(density))
    r_optim = Optim.MaximizationWrapper(_run_optim(f, x_init, algorithm))
    mode_trafo_unshaped = Optim.minimizer(r_optim.res)
    mode_trafo = shape(mode_trafo_unshaped)
    mode_notrafo = inv(trafo)(mode_trafo)
    (result = mode_notrafo, result_trafo = mode_trafo, trafo = trafo, info = r_optim)
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
