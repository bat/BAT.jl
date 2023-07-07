# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATUltraNestExt

@static if isdefined(Base, :get_extension)
    using UltraNest
else
    using ..UltraNest
end

using BAT

BAT.pkgext(::Val{:UltraNest}) = BAT.PackageExtension{:UltraNest}()

using BAT: AnyMeasureLike
using BAT: transform_and_unshape, all_active_names, exec_map!

using Random
using ArraysOfArrays
using DensityInterface, InverseFunctions, ValueShapes
import Measurements


function BAT.bat_sample_impl(
    target::AnyMeasureLike,
    algorithm::ReactiveNestedSampling,
    ::BATContext
)
    orig_measure = convert(BATMeasure, target)
    transformed_measure, trafo = transform_and_unshape(algorithm.trafo, orig_measure)

    vs = varshape(transformed_measure)

    if !(BAT._get_deep_transformable_base(transformed_measure) isa BAT.StdMvUniform)
        throw(ArgumentError("ReactiveNestedSampling only supports (transformed) densities defined on the unit hypercube"))
    end

    LogDType = Float64

    function vec_ultranest_logpstr(V_rowwise::AbstractMatrix{<:Real})
        V = deepcopy(V_rowwise')
        logd = similar(V, LogDType, size(V,2))
        V_nested = nestedview(V)
        exec_map!(logdensityof(transformed_measure), algorithm.executor, logd, V_nested)
    end

    ndims = totalndof(vs)
    paramnames = all_active_names(varshape(orig_measure))

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
    
    r = convert(Dict{String, Any}, unest_result)

    unest_wsamples = convert(Dict{String, Any}, r["weighted_samples"])
    v_trafo_us = nestedview(convert(Matrix{Float64}, unest_wsamples["points"]'))
    logvals_trafo = convert(Vector{Float64}, unest_wsamples["logl"])
    weight = convert(Vector{Float64}, unest_wsamples["weights"])
    samples_trafo = DensitySampleVector(vs.(v_trafo_us), logvals_trafo, weight = weight)
    samples_notrafo = inverse(trafo).(samples_trafo) 

    uwv_trafo_us = nestedview(convert(Matrix{Float64}, r["samples"]'))
    uwlogvals_trafo = map(logdensityof(transformed_measure), uwv_trafo_us)
    uwsamples_trafo = DensitySampleVector(vs.(uwv_trafo_us), uwlogvals_trafo)
    uwsamples_notrafo = inverse(trafo).(uwsamples_trafo)

    logz = convert(BigFloat, r["logz"])::BigFloat
    logzerr = convert(BigFloat, r["logzerr"])::BigFloat
    logintegral = Measurements.measurement(logz, logzerr)

    ess = convert(Float64, r["ess"])

    return (
        result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo,
        uwresult = uwsamples_notrafo, uwresult_trafo = uwsamples_trafo,
        logintegral = logintegral, ess = ess,
        info = r
    )
end


end # module BATUltraNestExt
