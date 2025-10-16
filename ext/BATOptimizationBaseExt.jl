# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationBaseExt

import OptimizationBase

using BAT
BAT.pkgext(::Val{:OptimizationBase}) = BAT.PackageExtension{:OptimizationBase}()

using BAT: MeasureLike, unevaluated
using BAT: get_context, get_adselector
using BAT: bat_initval, transform_and_unshape, apply_trafo_to_init

using DensityInterface, InverseFunctions, FunctionChains
using AutoDiffOperators: AbstractADType, NoAutoDiff, reverse_adtype


AbstractModeEstimator(optalg::Any) = OptimizationAlg(optalg)
Base.convert(::Type{AbstractModeEstimator}, alg::OptimizationAlg) = alg.optalg

BAT.ext_default(::BAT.PackageExtension{:OptimizationBase}, ::Val{:DEFAULT_OPTALG}) = nothing #Optim.NelderMead()


struct _OptimizationTargetFunc{F} <: Function
    f::F
end
_OptimizationTargetFunc(::Type{F}) where F = _OptimizationTargetFunc{Type{F}}(F)

(ft::_OptimizationTargetFunc)(x, ::Any) = ft.f(x)


build_optimizationfunction(f, ad::AbstractADType) = OptimizationBase.OptimizationFunction(f, ad)
build_optimizationfunction(f, ::NoAutoDiff) = OptimizationBase.OptimizationFunction(f)


function BAT.bat_findmode_impl(target::MeasureLike, algorithm::OptimizationAlg, context::BATContext)
    transformed_m, f_pretransform = transform_and_unshape(algorithm.pretransform, target, context)
    target_uneval = unevaluated(target)
    inv_trafo = inverse(f_pretransform)

    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_m, initalg, context).result)

    # Maximize density of original target, but run in transformed space, don't apply LADJ:
    f = fchain(inv_trafo, logdensityof(target_uneval), -)

    f_target = _OptimizationTargetFunc(f)
    ad = reverse_adtype(get_adselector(context))
    optimization_function = build_optimizationfunction(f_target, ad)
    optimization_problem = OptimizationBase.OptimizationProblem(optimization_function, x_init)

    algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    # Not all algorithms support abstol, just filter all NaN-valued opts out:
    filtered_algopts = NamedTuple(filter(p -> !isnan(p[2]), pairs(algopts)))
    optimization_result = OptimizationBase.solve(optimization_problem, algorithm.optalg; filtered_algopts..., algorithm.kwargs...) 

    transformed_mode =  optimization_result.u
    result_mode = inv_trafo(transformed_mode)

    (result = result_mode, result_trafo = transformed_mode, f_pretransform = f_pretransform, info = optimization_result)
end


end # module BATOptimizationBaseExt
