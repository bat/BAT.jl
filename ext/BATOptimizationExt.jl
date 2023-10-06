# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationExt

@static if isdefined(Base, :get_extension)
    import Optimization
else
    import ..Optimization
end

using BAT
BAT.pkgext(::Val{:Optimization}) = BAT.PackageExtension{:Optimization}()


using Random
using DensityInterface, ChangesOfVariables, InverseFunctions, FunctionChains
using HeterogeneousComputing, AutoDiffOperators
using StructArrays, ArraysOfArrays, ADTypes

using BAT: AnyMeasureOrDensity, AbstractMeasureOrDensity

using BAT: get_context, get_adselector, _NoADSelected
using BAT: bat_initval, transform_and_unshape, apply_trafo_to_init
using BAT: negative


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


function BAT.bat_findmode_impl(target::AnyMeasureOrDensity, algorithm::OptimizationAlg, context::BATContext)
    transformed_density, trafo = transform_and_unshape(algorithm.trafo, target, context)
    inv_trafo = inverse(trafo)

    initalg = apply_trafo_to_init(trafo, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)

    # Maximize density of original target, but run in transformed space, don't apply LADJ:
    f = fchain(inv_trafo, logdensityof(target), -)
    f2 = (x, p) -> f(x)

    adsel = get_adselector(context)

    optimization_function = build_optimizationfunction(f2, adsel)
    optimization_problem = Optimization.OptimizationProblem(optimization_function, x_init)

    algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    optimization_result = Optimization.solve(optimization_problem, algorithm.optalg; algopts..., algorithm.kwargs...) 

    transformed_mode =  optimization_result.u
    result_mode = inv_trafo(transformed_mode)

    (result = result_mode, result_trafo = transformed_mode, trafo = trafo, info = optimization_result)
end



end # module BATOptimizationExt
