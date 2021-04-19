# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# UltraNest docstrings are reproduced here under MIT License with the kind
# permission of the original author
# Johannes Buchner <johannes.buchner.acad@gmx.com>.


"""
    struct ReactiveNestedSampling <: AbstractUltraNestAlgorithm

*Experimental feature, not part of stable public API.*

[UltraNest](https://github.com/JohannesBuchner/UltraNest) reactive nested
sampling algorithm with.

Uses the UltraNest Python package, via
[UltraNest.jl](https://github.com/bat/UltraNest.jl) (and PyCall).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)


!!! note

    This functionality is only available when the
    [UltraNest](https://github.com/bat/UltraNest.jl) package is loaded (e.g. via
    `import UltraNest`).
"""
@with_kw struct ReactiveNestedSampling{TR<:AbstractDensityTransformTarget} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()

    # "Indicating whether this parameter wraps around (circular parameter)"
    # wrapped_params::Array{Bool}

    "Test transform and likelihood with this number of random points for errors first. Useful to catch bugs."
    num_test_samples::Int = 2

    "If efficiency goes down, dynamically draw more points from the region between ndraw_min and ndraw_max. If set to False, few points are sampled at once."
    draw_multiple::Bool = true

    "Number of logZ estimators and MLFriends region bootstrap rounds."
    num_bootstraps::Int = 30

    "Minimum number of points to simultaneously propose. Increase this if your likelihood makes vectorization very cheap."
    ndraw_min::Int = 128

    "Maximum number of points to simultaneously propose. Increase this if your likelihood makes vectorization very cheap. Memory allocation may be slow for extremely high values."
    ndraw_max::Int = 65536

    "Update region when the volume shrunk by this amount."
    update_interval_volume_fraction::Float64 = 0.8

    "Update stdout status line every log_interval iterations."
    log_interval::Int = -1
    
    "Show integration progress as a status line."
    show_status::Bool = true
    
    # "Callback function when region was rebuilt. Allows to show current state of the live points."
    # viz_callback::Function = nop_func

    "Target evidence uncertainty. This is the std between bootstrapped logz integrators."
    dlogz::Float64 = 0.5
    
    "Target posterior uncertainty. This is the Kullback-Leibler divergence in nat between bootstrapped integrators."
    dKL::Float64 = 0.5
    
    "Integrate until this fraction of the integral is left in the remainder. Set to a low number (1e-2 … 1e-5) to make sure peaks are discovered. Set to a higher number (0.5) if you know the posterior is simple."
    frac_remain::Float64 = 0.01
    
    "Terminate when live point likelihoods are all the same, within Lepsilon tolerance. Increase this when your likelihood function is inaccurate, to avoid unnecessary search."
    Lepsilon::Float64 = 0.001
    
    "Target number of effective posterior samples."
    min_ess::Int = 400
    
    "maximum number of integration iterations."
    max_iters::Int = -1
    
    "Stop after this many likelihood evaluations."
    max_ncalls::Int = -1
    
    "The algorithm tries to assess iteratively where more samples are needed. This number limits the number of improvement loops."
    max_num_improvement_loops::Int = -1
    
    "Minimum number of live points throughout the run."
    min_num_live_points::Int = 400
    
    "Require at least this many live points per detected cluster."
    cluster_num_live_points::Int = 40
    
    "z-score used as a threshold for the insertion order test. Set to infinity to disable."
    insertion_test_window::Float64 = 10.0
    
    "Number of iterations after which the insertion order test is reset."
    insertion_test_zscore_threshold::Float64 = 2.0
end
export ReactiveNestedSampling


function bat_sample_impl(
    rng::AbstractRNG,
    target::AnyDensityLike,
    algorithm::ReactiveNestedSampling
)
    density_notrafo = convert(AbstractDensity, target)
    shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
    vs = varshape(shaped_density)
    density = unshaped(shaped_density)

    bounds = var_bounds(density)
    if !(all(isequal(0), bounds.vol.lo) && all(isequal(1), bounds.vol.hi))
        throw(ArgumentError("ReactiveNestedSampling only supports (transformed) densities defined on the unit hypercube"))
    end

    function vec_ultranest_logpstr(V_rowwise::AbstractMatrix{<:Real})
        map(logdensityof(density), nestedview(copy(V_rowwise')))
    end

    ndims = totalndof(vs)
    paramnames = all_active_names(varshape(density_notrafo))

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
        #viz_callback = algorithm.# viz_callback,
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
    samples_notrafo = inv(trafo).(samples_trafo)

    uwv_trafo_us = nestedview(convert(Matrix{Float64}, r["samples"]'))
    uwlogvals_trafo = map(logdensityof(density), uwv_trafo_us)
    uwsamples_trafo = DensitySampleVector(vs.(uwv_trafo_us), uwlogvals_trafo)
    uwsamples_notrafo = inv(trafo).(uwsamples_trafo)

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
