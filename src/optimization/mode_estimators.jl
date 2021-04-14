# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ModeAsDefined <: AbstractModeEstimator

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.

Constructors:

* ```$(FUNCTIONNAME)()```        
"""
struct ModeAsDefined <: AbstractModeEstimator end
export ModeAsDefined


function bat_findmode_impl(target::AnySampleable, algorithm::ModeAsDefined)
    (result = StatsBase.mode(target),)
end

function bat_findmode_impl(target::Distribution, algorithm::ModeAsDefined)
    (result = varshape(target)(StatsBase.mode(unshaped(target))),)
end

function bat_findmode_impl(target::DistributionDensity, algorithm::ModeAsDefined)
    bat_findmode_impl(parent(target), algorithm)
end



"""
    MaxDensitySampleSearch <: AbstractModeEstimator

Constructors:

    MaxDensitySampleSearch()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct MaxDensitySampleSearch <: AbstractModeEstimator end
export MaxDensitySampleSearch


function bat_findmode_impl(target::DensitySampleVector, algorithm::MaxDensitySampleSearch)
    v, i = _get_mode(target)
    (result = v, mode_idx = i)
end



"""
    struct MaxDensityNelderMead <: AbstractModeEstimator

Estimates the mode of a probability density using Nelder-Mead optimization
(currently via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
subject to change).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MaxDensityNelderMead{
    TR<:AbstractDensityTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    trafo::TR = NoDensityTransform()
    init::IA = InitFromTarget()
end
export MaxDensityNelderMead

function bat_findmode_impl(target::AnySampleable, algorithm::MaxDensityNelderMead)
    density_notrafo = convert(AbstractDensity, target)
    shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
    shape = varshape(shaped_density)
    density = unshaped(shaped_density)

    rng = bat_determ_rng()
    x = collect(unshaped(bat_initval(rng, density, apply_trafo_to_init(trafo, algorithm.init)).result))
    r_optim = Optim.maximize(p -> eval_logval(density, p, default_dlt()), x, Optim.NelderMead())
    mode_trafo = shape(Optim.minimizer(r_optim.res))
    mode_notrafo = inv(trafo)(mode_trafo)
    (result = mode_notrafo, result_trafo = mode_trafo, trafo = trafo, info = r_optim)
end


"""
    struct MaxDensityLBFGS <: AbstractModeEstimator

Estimates the mode of a probability density using LBFGS optimization
(currently via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl),
subject to change).

The gradient of the target density is computed via auto-differentiation.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MaxDensityLBFGS{
    TR<:AbstractDensityTransformTarget,
    IA<:InitvalAlgorithm
} <: AbstractModeEstimator
    trafo::TR = PriorToGaussian()
    init::IA = InitFromTarget()
end
export MaxDensityLBFGS


function bat_findmode_impl(target::AnySampleable, algorithm::MaxDensityLBFGS)
    density_notrafo = convert(AbstractDensity, target)
    shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
    shape = varshape(shaped_density)
    density = unshaped(shaped_density)

    rng = bat_determ_rng()
    x = collect(unshaped(bat_initval(rng, density, apply_trafo_to_init(trafo, algorithm.init)).result))
    r_optim = Optim.maximize(p -> eval_logval(density, p, default_dlt()), x, Optim.LBFGS(); autodiff = :forward)
    mode_trafo = shape(Optim.minimizer(r_optim.res))
    mode_notrafo = inv(trafo)(mode_trafo)
    (result = mode_notrafo, result_trafo = mode_trafo, trafo = trafo, info = r_optim)
end



function bat_marginalmode_impl(samples::DensitySampleVector; nbins::Union{Integer, Symbol} = 200)

    shape = varshape(samples)
    flat_samples = flatview(unshaped.(samples.v))
    n_params = size(flat_samples)[1]
    nt_samples = ntuple(i -> flat_samples[i,:], n_params)
    marginalmode_params = Vector{Float64}()

    for param in Base.OneTo(n_params)
        if typeof(nbins) == Symbol
            number_of_bins = _auto_binning_nbins(nt_samples, param, mode=nbins)
        else
            number_of_bins = nbins
        end

        marginalmode_param = find_marginalmodes(get_marginal_dist(samples, param, bins=number_of_bins).result)

        if length(marginalmode_param[1]) > 1
            @warn "More than one bin with the same weight is found. Returned the first one"
        end
        push!(marginalmode_params, marginalmode_param[1][1])
    end
    (result = shape(marginalmode_params),)
end


# From Plots.jl, original authors Oliver Schulz and Michael K. Borregaard
function _auto_binning_nbins(vs::NTuple{N,AbstractVector}, dim::Integer; mode::Symbol = :auto) where N
    max_bins = 10_000
    _cl(x) = min(ceil(Int, max(x, one(x))), max_bins)
    _iqr(v) = (q = quantile(v, 0.75) - quantile(v, 0.25); q > 0 ? q : oftype(q, 1))
    _span(v) = maximum(v) - minimum(v)

    n_samples = length(LinearIndices(first(vs)))

    # The nd estimator is the key to most automatic binning methods, and is modified for twodimensional histograms to include correlation
    nd = n_samples^(1/(2+N))
    nd = N == 2 ? min(n_samples^(1/(2+N)), nd / (1-cor(first(vs), last(vs))^2)^(3//8)) : nd # the >2-dimensional case does not have a nice solution to correlations

    v = vs[dim]

    if mode == :auto
        mode = :fd
    end

    if mode == :sqrt  # Square-root choice
        _cl(sqrt(n_samples))
    elseif mode == :sturges  # Sturges' formula
        _cl(log2(n_samples) + 1)
    elseif mode == :rice  # Rice Rule
        _cl(2 * nd)
    elseif mode == :scott  # Scott's normal reference rule
        _cl(_span(v) / (3.5 * std(v) / nd))
    elseif mode == :fd  # Freedman–Diaconis rule
        _cl(_span(v) / (2 * _iqr(v) / nd))
    elseif mode == :wand
        _cl(wand_edges(v))  # this makes this function not type stable, but the type instability does not propagate
    else
        error("Unknown auto-binning mode $mode")
    end
end
