# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    ModeOptimization

Find the mode of a probability measure/distribution by using an optimization
algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

`optimalg` must be an `Optim.AbstractOptimizer`.

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct ModeOptimization{
    ALG,
    TR<:AbstractTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    optimizer::ALG = #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pretransform::TR = PriorToNormal()
    init::IA = InitFromTarget()
end
export ModeOptimization


function BAT.bat_findmode_impl(target::MeasureLike, algorithm::ModeOptimization, context::BATContext)
    transformed_density, f_pretransform = transform_and_unshape(algorithm.pretransform, target, context)
    target_uneval = unevaluated(target)
    inv_trafo = inverse(f_pretransform)

    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)

    # Maximize density of original target, but run in transformed space, don't apply LADJ:
    f = fchain(inv_trafo, logdensityof(target_uneval), -)

    r = bat_maximize_impl(f, x_init, algorithm.optimizer, context)

    transformed_mode = r.result
    result_mode = inv_trafo(transformed_mode)

    return (
        result = result_mode, result_trafo = transformed_mode, f_pretransform = f_pretransform,
        #=trace_trafo = trace_trafo,=#
        info = r.info
    )
end
