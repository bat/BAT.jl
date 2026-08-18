# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# TODO (breaking): Disentangle MaxDensityAlgorithm from AbstractModeEstimator,
# create a mode estimator algorithm that pairs a MaxDensityAlgorithm with a
# pretransform:
const MaxDensityAlgorithm = Union{OptimAlg, OptimizationAlg}


function evalmeasure_impl(measure::BATMeasure, algorithm::MaxDensityAlgorithm, context::BATContext)
    # Maximize the density of the original measure, searching in the
    # reparametrized space without applying its LADJ:
    transformed_density, f_pretransform = transform_and_unshape(algorithm.pretransform, measure, context)
    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)
    f = fchain(inverse(f_pretransform), checked_logdensityof(unevaluated(measure)))
    r = maximize_density(f, x_init, algorithm, context)
    o = _optimum_result(r, f_pretransform)
    return EvalMeasureImplReturn(;
        modes = [o.result],
        evalresult = Base.structdiff(o, NamedTuple{(:result,)})
    )
end

function bat_bgml_impl(likelihood, prior, algorithm::MaxDensityAlgorithm, context::BATContext)
    # Maximize the likelihood only:
    pr = batmeasure(prior)
    li = _convert_likelihood(likelihood, DensityKind(likelihood))
    transformed_pr, f_pretransform = transform_and_unshape(algorithm.pretransform, pr, context)
    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_pr, initalg, context).result)
    f = fchain(inverse(f_pretransform), checked_logdensityof(li))
    r = maximize_density(f, x_init, algorithm, context)
    return _optimum_result(r, f_pretransform)
end
