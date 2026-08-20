# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimizationBaseExt

import OptimizationBase

using BAT
BAT.pkgext(::Val{:OptimizationBase}) = BAT.PackageExtension{:OptimizationBase}()

using BAT: get_adselector

using FunctionChains
using AutoDiffOperators: AbstractADType, NoAutoDiff, reverse_adtype


BAT.ext_default(::BAT.PackageExtension{:OptimizationBase}, ::Val{:DEFAULT_OPTALG}) = nothing #Optim.NelderMead()


struct _OptimizationTargetFunc{F} <: Function
    f::F
end
_OptimizationTargetFunc(::Type{F}) where F = _OptimizationTargetFunc{Type{F}}(F)

(ft::_OptimizationTargetFunc)(x, ::Any) = ft.f(x)


build_optimizationfunction(f, ad::AbstractADType) = OptimizationBase.OptimizationFunction(f, ad)
build_optimizationfunction(f, ::NoAutoDiff) = OptimizationBase.OptimizationFunction(f)


function BAT.maximize_density(f_logdensity, x_init::AbstractVector{<:Real}, algorithm::OptimizationAlg, context::BATContext)
    f = fchain(f_logdensity, -)

    f_target = _OptimizationTargetFunc(f)
    ad = reverse_adtype(get_adselector(context))
    optimization_function = build_optimizationfunction(f_target, ad)
    optimization_problem = OptimizationBase.OptimizationProblem(optimization_function, x_init)

    algopts = (maxiters = algorithm.maxiters, maxtime = algorithm.maxtime, abstol = algorithm.abstol, reltol = algorithm.reltol)
    # Not all algorithms support abstol, just filter all NaN-valued opts out:
    filtered_algopts = NamedTuple(filter(p -> !isnan(p[2]), pairs(algopts)))
    optimization_result = OptimizationBase.solve(optimization_problem, algorithm.optalg; filtered_algopts..., algorithm.kwargs...)

    (result = optimization_result.u, info = optimization_result)
end


end # module BATOptimizationBaseExt
