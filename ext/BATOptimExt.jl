# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATOptimExt

using Optim: Optim, OnceDifferentiable

using BAT
BAT.pkgext(::Val{:Optim}) = BAT.PackageExtension{:Optim}()

using BAT: MeasureLike, BATMeasure, unevaluated
using BAT: get_context, get_valid_adselector
using BAT: bat_initval, transform_and_unshape, apply_trafo_to_init

using DensityInterface, InverseFunctions, FunctionChains
using StructArrays, ArraysOfArrays
using AutoDiffOperators: reverse_adtype


AbstractModeEstimator(optalg::Optim.AbstractOptimizer) = OptimAlg(optalg)
Base.convert(::Type{AbstractModeEstimator}, alg::OptimAlg) = alg.optalg

BAT.ext_default(::BAT.PackageExtension{:Optim}, ::Val{:DEFAULT_OPTALG}) = Optim.NelderMead()
BAT.ext_default(::BAT.PackageExtension{:Optim}, ::Val{:NELDERMEAD_ALG}) = Optim.NelderMead()
BAT.ext_default(::BAT.PackageExtension{:Optim}, ::Val{:LBFGS_ALG}) = Optim.LBFGS()

function convert_options(algorithm::OptimAlg)
    if !isnan(algorithm.abstol)
       @warn "The option 'abstol=$(algorithm.abstol)' is not used for this algorithm."
    end

    kwargs = algorithm.kwargs

    algopts = (; iterations = algorithm.maxiters, time_limit = algorithm.maxtime, f_reltol = algorithm.reltol,)
    algopts = (; algopts..., kwargs...)
    algopts = (; algopts..., store_trace = true, extended_trace=true) 

    return Optim.Options(; algopts...)
end 

function BAT.bat_findmode_impl(target::MeasureLike, algorithm::OptimAlg, context::BATContext)
    transformed_density, f_pretransform = transform_and_unshape(algorithm.pretransform, target, context)
    target_uneval = unevaluated(target)
    inv_trafo = inverse(f_pretransform)

    initalg = apply_trafo_to_init(f_pretransform, algorithm.init)
    x_init = collect(bat_initval(transformed_density, initalg, context).result)

    # Maximize density of original target, but run in transformed space, don't apply LADJ:
    f = fchain(inv_trafo, logdensityof(target_uneval), -)
    opts = convert_options(algorithm)
    optim_result = _optim_minimize(f, x_init, algorithm.optalg, opts, context)
    r_optim = Optim.MaximizationWrapper(optim_result)
    transformed_mode = Optim.minimizer(r_optim.res)
    result_mode = inv_trafo(transformed_mode)

    # ToDo: Re-enable trace, make it type stable:
    #dummy_f_x = f(x_init) # ToDo: Avoid recomputation
    #trace_trafo = StructArray(;_neg_opt_trace(optim_result, x_init, dummy_f_x) ...)

    ret_a = (result = result_mode, result_trafo = transformed_mode, f_pretransform = f_pretransform #=trace_trafo = trace_trafo=#)
    ret_b = @NamedTuple{info::Optim.MaximizationWrapper}((r_optim,))
    return merge(ret_a, ret_b)
end

function _optim_minimize(f::Function, x_init::AbstractArray{<:Real}, algorithm::Optim.ZerothOrderOptimizer, opts::Optim.Options, ::BATContext)
    _optim_optimize(f, x_init, algorithm, opts)
end

function _optim_minimize(f::Function, x_init::AbstractArray{<:Real}, algorithm::Optim.FirstOrderOptimizer, opts::Optim.Options, context::BATContext)
    ad = reverse_adtype(get_valid_adselector(context, algorithm))
    target = OnceDifferentiable(f, x_init, autodiff = ad)
    _optim_optimize(target, x_init, algorithm, opts)
end

# Wrapper for type stability of optimize result (why does this work?):
function _optim_optimize(target::TRG, x0::AbstractArray, method::Optim.AbstractOptimizer, options = Optim.Options()) where TRG
    Optim.optimize(target, x0, method, options)
end


function _neg_opt_trace(
    @nospecialize(optim_result::Optim.MultivariateOptimizationResults),
    dummy_x::AbstractVector{<:Real}, dummy_f_x::Real
)
    trc = Optim.trace(optim_result)
    tr_len = length(eachindex(trc))
    nd = length(eachindex(dummy_x))

    v = nestedview(similar(dummy_x, nd, tr_len))
    foreach((a,b) -> a[:] = b, v, Optim.x_trace(optim_result))

    logd = similar(dummy_x, typeof(dummy_f_x), tr_len)
    logd[:] = - Optim.f_trace(optim_result)

    if optim_result isa Optim.MultivariateOptimizationResults{<:Optim.ZerothOrderOptimizer}
        (v = v, logd = logd)
    else
        grad_logd = nestedview(similar(dummy_x, nd, tr_len))
        foreach((a,b) -> a[:] = -b.metadata["g(x)"], grad_logd, trc)
        (v = v, logd = logd, grad_logd = grad_logd)
    end
end

function _neg_opt_trace(
    @nospecialize(optim_result::Optim.MultivariateOptimizationResults{<:Optim.NelderMead}),
    dummy_x::AbstractVector{<:Real}, dummy_f_x::Real
)
    trc = Optim.trace(optim_result)
    tr_len = length(eachindex(trc))
    nd = length(eachindex(dummy_x))

    v = nestedview(similar(dummy_x, nd, tr_len))
    foreach((a,b) -> a[:] = b, v, Optim.centroid_trace(optim_result))

    (;v = v)
end


end # module BATOptimExt
