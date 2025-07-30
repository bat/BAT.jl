# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATUltraNestExt

using UltraNest

using BAT

BAT.pkgext(::Val{:UltraNest}) = BAT.PackageExtension{:UltraNest}()

using BAT: MeasureLike, BATMeasure
using BAT: transform_and_unshape, measure_support, all_active_names, exec_map!

using Random
using ArraysOfArrays
using DensityInterface, InverseFunctions, ValueShapes
import Measurements


function BAT.evalmeasure_impl(measure::BATMeasure, algorithm::ReactiveNestedSampling, context::BATContext)
    m = unevaluated(measure)
    transformed_m, f_pretransform = transform_and_unshape(algorithm.pretransform, m, context)
    n_dof = some_dof(transformed_m)

    if !BAT.has_uhc_support(transformed_m)
        throw(ArgumentError("$algorithm doesn't measures that are not limited to the unit hypercube"))
    end

    LogDType = Float64

    function vec_ultranest_logpstr(V_rowwise::AbstractMatrix{<:Real})
        V = deepcopy(V_rowwise')
        logd = similar(V, LogDType, size(V,2))
        V_nested = nestedview(V)
        exec_map!(logdensityof(transformed_m), algorithm.executor, logd, V_nested)
    end

    paramnames = all_active_names(varshape(m))

    ch = Channel()
    function run_sampler()
        try
            smplr = UltraNest.ultranest.ReactiveNestedSampler(
                paramnames, vec_ultranest_logpstr, vectorized = true,
                num_test_samples = algorithm.num_test_samples,
                draw_multiple = algorithm.draw_multiple,
                num_bootstraps = algorithm.num_bootstraps,
                ndraw_min = algorithm.ndraw_min,
                ndraw_max = algorithm.ndraw_max
            )

            unest_result = smplr.run(
                log_interval = algorithm.log_interval < 0 ? nothing : algorithm.log_interval,
                show_status = algorithm.show_status,
                viz_callback = algorithm.viz_callback,
                dlogz = algorithm.dlogz,
                dKL = algorithm.dKL,
                frac_remain = algorithm.frac_remain,
                Lepsilon = algorithm.Lepsilon,
                min_ess = algorithm.min_ess,
                max_iters = algorithm.max_iters < 0 ? nothing : algorithm.max_iters,
                max_ncalls = algorithm.max_ncalls < 0 ? nothing : algorithm.max_ncalls,
                max_num_improvement_loops = algorithm.max_num_improvement_loops,
                min_num_live_points = algorithm.min_num_live_points,
                cluster_num_live_points = algorithm.cluster_num_live_points,
                insertion_test_window = algorithm.insertion_test_window,
                insertion_test_zscore_threshold = algorithm.insertion_test_zscore_threshold
            )
            put!(ch, unest_result)
        finally
            close(ch)
        end
    end

    # Force Python interaction to run on thread 1:
    task_id = 1
    task = Task(run_sampler)
    task.sticky = true
    # From ThreadPools.@tspawnat:
    ccall(:jl_set_task_tid, Cvoid, (Any, Cint), task, task_id-1)
    schedule(task)
    unest_result = take!(ch)

    r = convert(Dict{String, Any}, unest_result)

    unest_wsamples = convert(Dict{String, Any}, r["weighted_samples"])
    v_trafo_us = nestedview(convert(Matrix{Float64}, unest_wsamples["points"]'))
    logvals_trafo = convert(Vector{Float64}, unest_wsamples["logl"])
    weight = convert(Vector{Float64}, unest_wsamples["weights"])
    transformed_smpls = DensitySampleVector(v = v_trafo_us, logd = logvals_trafo, weight = weight)
    smpls = inverse(f_pretransform).(transformed_smpls) 

    uwv_trafo_us = nestedview(convert(Matrix{Float64}, r["samples"]'))
    uwlogvals_trafo = map(logdensityof(transformed_m), uwv_trafo_us)
    uwtransformed_smpls = DensitySampleVector(uwv_trafo_us, uwlogvals_trafo)
    uwsmpls = inverse(f_pretransform).(uwtransformed_smpls)

    logz = convert(BigFloat, r["logz"])::BigFloat
    logzerr = convert(BigFloat, r["logzerr"])::BigFloat
    mass = exp(ULogarithmic, Measurements.measurement(logz, logzerr))

    ess = convert(Float64, r["ess"])

    return (
        result_trafo = transformed_smpls, f_pretransform = f_pretransform,
        uwresult = uwsmpls, uwresult_trafo = uwtransformed_smpls,
        ultranest_result = r
    )

    dsm = DensitySampleMeasure(smpls, dof = n_dof, ess = ess, mass = mass)

    return EvalMeasureImplReturn(;
        empirical = dsm,
        dof = n_dof,
        mass = mass,
        # ToDo:
        # modes = ...,
        evalresult = evalresult
    )
end


end # module BATUltraNestExt
