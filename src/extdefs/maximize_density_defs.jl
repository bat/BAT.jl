# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const MaxDensityAlgorithm = Union{OptimAlg, OptimizationAlg}


# Pathfinder needs a gradient-based optimizer whose maximize_density
# implementation records a gradient trace. Kept as a function so that
# PathfinderTransformInit construction gives a clear error when no
# suitable backend is available:
function _default_pathfinder_optalg()
    try
        return OptimAlg(optalg = ext_default(pkgext(Val(:Optim)), Val(:LBFGS_ALG)))
    catch err
        err isa ErrorException || rethrow()
        throw(ErrorException(
            "The default PathfinderTransformInit backend requires the Optim package to be loaded. Load Optim, or set the optalg field explicitly to a gradient-based backend that supports trace recording (e.g. an OptimizationAlg with an L-BFGS solver)."
        ))
    end
end


"""
    struct TransformedMaxDensity <: AbstractModeEstimator

Estimates the mode of a measure by maximizing its density numerically,
searching in a transformed space.

The search runs in the space induced by `pretransform` without applying
the transformation's volume correction, so the result is a mode of the
original density, not of the transformed one.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedMaxDensity{
    A<:MaxDensityAlgorithm,
    TR<:TransformIntent,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    "Density maximization backend."
    optalg::A = OptimAlg()
    "Target space transformation to search in."
    pretransform::TR = NormalBased()
    "Initial point selection, applied in the transformed space."
    init::IA = InitFromTarget()
end
export TransformedMaxDensity

batalgorithm(optalg::MaxDensityAlgorithm) = TransformedMaxDensity(optalg = optalg)


function evalmeasure_impl(em::EvaluatedMeasure, algorithm::TransformedMaxDensity, context::BATContext)
    # Maximize the density of the original measure, searching in the
    # reparametrized space without applying its LADJ:
    transformed_density, f_pretransform = transform_and_unshape(algorithm.pretransform, em, context)
    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)
    f = fchain(inverse(f_pretransform), checked_logdensityof(unevaluated(em)))
    r = maximize_density(f, x_init, algorithm.optalg, context)
    o = _optimum_result(r, f_pretransform)
    return EvaluatedMeasure(em;
        modes = [o.result],
        evalinfo = MeasureEvalInfo(algorithm, Base.structdiff(o, NamedTuple{(:result,)}))
    )
end

function bat_bgml_impl(likelihood, prior, algorithm::TransformedMaxDensity, context::BATContext)
    # Maximize the likelihood only:
    pr = batmeasure(prior)
    li = _convert_likelihood(likelihood, DensityKind(likelihood))
    transformed_pr, f_pretransform = transform_and_unshape(algorithm.pretransform, pr, context)
    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_pr, initalg, context).result)
    f = fchain(inverse(f_pretransform), checked_logdensityof(li))
    r = maximize_density(f, x_init, algorithm.optalg, context)
    return _optimum_result(r, f_pretransform)
end
