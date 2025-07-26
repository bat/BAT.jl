# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationExt

import Optimization

using BAT
BAT.pkgext(::Val{:Optimization}) = BAT.PackageExtension{:Optimization}()


using Random
using DensityInterface, ChangesOfVariables, InverseFunctions, FunctionChains
using HeterogeneousComputing, AutoDiffOperators
using StructArrays, ArraysOfArrays, ADTypes

using BAT: MeasureLike, unevaluated

using BAT: get_context, get_adselector, _NoADSelected
using BAT: bat_initval, transform_and_unshape, apply_trafo_to_init
# using BAT: negative #deprecated? 

function test_bat_optimization_ext()
    println("BAT_Optimization_Ext is included")
end

AbstractModeEstimator(optalg::Any) = OptimizationAlg(optalg)
Base.convert(::Type{AbstractModeEstimator}, alg::OptimizationAlg) = alg.optalg

BAT.ext_default(::BAT.PackageExtension{:Optimization}, ::Val{:DEFAULT_OPTALG}) = nothing #Optim.NelderMead()


function build_optimizationfunction(f, adsel::AutoDiffOperators.ADSelector)
    adm = convert(ADTypes.AbstractADType, reverse_ad_selector(adsel))
    optimization_function = Optimization.OptimizationFunction(f, adm)
    return optimization_function
end

function build_optimizationfunction(f, adsel::BAT._NoADSelected)
    optimization_function = Optimization.OptimizationFunction(f)
    return optimization_function
end


function evalmeasure_impl(target::BATMeasure, algorithm::OptimizationAlg, context::BATContext)
    transformed_m, f_pretransform = transform_and_unshape(algorithm.pretransform, target, context)
    target_uneval = unevaluated(target)
    inv_trafo = inverse(f_pretransform)

    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_m, initalg, context).result)

    # Maximize density of original target, but run in transformed space, don't apply LADJ:
    f = fchain(inv_trafo, logdensityof(target_uneval), -)
    target_f = (x, p) -> f(x)

    adsel = get_adselector(context)

    optimization_function = build_optimizationfunction(target_f, adsel)
    optimization_problem = Optimization.OptimizationProblem(optimization_function, x_init)

    algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    # Not all algorithms support abstol, just filter all NaN-valued opts out:
    filtered_algopts = NamedTuple(filter(p -> !isnan(p[2]), pairs(algopts)))
    optimization_result = Optimization.solve(optimization_problem, algorithm.optalg; filtered_algopts..., algorithm.kwargs...) 

    transformed_mode =  optimization_result.u
    result_mode = inv_trafo(transformed_mode)

    EvalMeasureImplReturn(;
        modes = [result_mode],
        evalresult = (result_trafo = transformed_mode, f_pretransform = f_pretransform, info = optimization_result),
    )
end


end # module BATOptimizationExt
