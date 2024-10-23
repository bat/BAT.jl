# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationExt

import Optimization

using BAT
BAT.pkgext(::Val{:Optimization}) = BAT.PackageExtension{:Optimization}()


using Random
using DensityInterface, ChangesOfVariables, InverseFunctions, FunctionChains
using HeterogeneousComputing, AutoDiffOperators
using StructArrays, ArraysOfArrays, ADTypes

using BAT: MeasureLike

using BAT: get_context, get_adselector, _NoADSelected
using BAT: bat_initval, transform_and_unshape, apply_trafo_to_init
# using BAT: negative #deprecated? 

function test_bat_optimization_ext()
    println("BAT_Optimization_Ext is included")
end

AbstractModeEstimator(optalg::Any) = OptimizationAlg(optalg)
convert(::Type{AbstractModeEstimator}, alg::OptimizationAlg) = alg.optalg

BAT.ext_default(::BAT.PackageExtension{:Optimization}, ::Val{:DEFAULT_OPTALG}) = nothing #Optim.NelderMead()


function build_optimizationfunction(f, adsel::AutoDiffOperators.ADSelector)
    adm = convert_ad(ADTypes.AbstractADType, adsel)
    optimization_function = Optimization.OptimizationFunction(f, adm)
    return optimization_function
end

function build_optimizationfunction(f, adsel::BAT._NoADSelected)
    optimization_function = Optimization.OptimizationFunction(f)
    return optimization_function
end


function BAT.bat_findmode_impl(target::MeasureLike, algorithm::OptimizationAlg, context::BATContext)
    transformed_density, f_pretransform = transform_and_unshape(algorithm.pretransform, target, context)
    inv_trafo = inverse(f_pretransform)

    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)

    # Maximize density of original target, but run in transformed space, don't apply LADJ:
    f = fchain(inv_trafo, logdensityof(target), -)
    target_f = (x, p) -> f(x)

    adsel = get_adselector(context)

    optimization_function = build_optimizationfunction(target_f, adsel)
    optimization_problem = Optimization.OptimizationProblem(optimization_function, x_init)

    algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    optimization_result = Optimization.solve(optimization_problem, algorithm.optalg; algopts..., algorithm.kwargs...) 

    transformed_mode =  optimization_result.u
    result_mode = inv_trafo(transformed_mode)

    (result = result_mode, result_trafo = transformed_mode, f_pretransform = f_pretransform, info = optimization_result)
end



end # module BATOptimizationExt
