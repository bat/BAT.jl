
"""
    struct SampledMeasure <: BATMeasure

Stores a density and samples drawn from it.

A report on the density, samples, properties of variates/parameters, etc. can
be generated via `Base.show`.

Constructors:

* ```SampledMeasure(density::AbstractPosteriorMeasure, samples::DensitySampleVector)```

Fields:

* ```density::BATMeasure```

* ```samples::DensitySamplesVector```

* ```_stats::BAT.MCMCBasicStats```

* ```_generator::AbstractSampleGenerator```

!!! note

    Fields `_stats` and `_generator` do not form part of the stable public
    API and are subject to change without deprecation.
"""
struct SampledMeasure{D<:BATMeasure,S<:DensitySampleVector, G<:AbstractSampleGenerator} <: BATMeasure
    density::D
    samples::S
    _stats::MCMCBasicStats
    _generator::G
end
export SampledMeasure


function SampledMeasure(density::BATMeasure, samples::DensitySampleVector)
    stats = MCMCBasicStats(samples)
    return SampledMeasure(density, samples, stats, UnknownSampleGenerator())
end

function SampledMeasure(density::SampledMeasure, samples::DensitySampleVector)
    return SampledMeasure(density.density, samples)
end

function SampledMeasure(target::AnyMeasureOrDensity, samples::DensitySampleVector)
    return SampledMeasure(convert(AbstractMeasureOrDensity, target), samples)
end


eval_logval(density::SampledMeasure, v::Any, T::Type{<:Real}) = eval_logval(density.density, v, T)

eval_logval_unchecked(density::SampledMeasure, v::Any) = eval_logval_unchecked(density.density, v)

ValueShapes.varshape(density::SampledMeasure) = varshape(density.density)

var_bounds(density::SampledMeasure) = var_bounds(density.density)


# ToDo: Distributions.sampler(density::SampledMeasure)
# ToDo: bat_sampler(density::SampledMeasure)


get_initsrc_from_target(target::SampledMeasure) = target.samples


_get_deep_prior_for_trafo(density::SampledMeasure) = _get_deep_prior_for_trafo(density.density)

function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::SampledMeasure, algorithm::PriorSubstitution)
    new_parent_density, trafo = bat_transform_impl(target, density.density, algorithm)
    new_samples = trafo.(density.samples)
    (result = SampledMeasure(new_parent_density, new_samples), trafo = trafo)
end

# ToDo: truncate_density(density::SampledMeasure, bounds::AbstractArray{<:Interval})

_approx_cov(target::SampledMeasure) = cov(target.samples)


function getdensity(sd::SampledMeasure)
    return sd.density
end

function nfreeparams(sd::SampledMeasure)
    return totalndof(varshape(sd))
end

function freeparams(sd::SampledMeasure)
    return active_keys(sd)
end

function nfixedparams(sd::SampledMeasure)
    return length(get_fixed_names(varshape(sd)))
end

function fixedparams(sd::SampledMeasure)
    param_shape = varshape(sd)
    fixed_param_keys = Symbol.(get_fixed_names(param_shape))
    fixed_values = [getproperty(param_shape, f).shape.value for f in fixed_param_keys]
    return (; zip(fixed_param_keys, fixed_values)...,)
end

function numberofsamples(sd::SampledMeasure)
    return length(sd.samples)
end

function eff_sample_size(sd::SampledMeasure)
    effsize = bat_eff_sample_size(unshaped.(sd.samples)).result
    return (; zip(active_keys(sd), effsize)...,)

end

function active_keys(sd::SampledMeasure)
    return Symbol.(all_active_names(elshape(sd.samples.v)))
end

function Statistics.mean(sd::SampledMeasure)
    means = sd._stats.param_stats.mean
    return (; zip(active_keys(sd), means)...,)
end

function Statistics.std(sd::SampledMeasure)
    covm = collect(sd._stats.param_stats.cov)
    stds = sqrt.(LinearAlgebra.diag(covm))

    return (; zip(active_keys(sd), stds)...,)
end

function Distributions.mode(sd::SampledMeasure)
    modes = sd._stats.mode
    return (; zip(active_keys(sd), modes)...,)
end

function marginalmode(sd::SampledMeasure)
    modes = bat_marginalmode(unshaped.(sd.samples)).result
    return (; zip(active_keys(sd), modes)...,)
end

function Statistics.cov(sd::SampledMeasure)
    covm = collect(sd._stats.param_stats.cov)
    names = string.(active_keys(sd))

    return NamedArrays.NamedArray(covm, (names, names), ("cov",""))
end

function Statistics.cor(sd::SampledMeasure)
    covm = collect(sd._stats.param_stats.cov)
    corm = cov2cor(covm, sqrt.(diag(covm)))
    names = string.(active_keys(sd))

    return NamedArrays.NamedArray(corm, (names, names), ("cor",""))
end


function parameter_table(sd::SampledMeasure)
    tab = TypedTables.Table(
        parameter = active_keys(sd),
        mean = collect(mean(sd)),
        std = collect(std(sd)),
        global_mode = collect(mode(sd)),
        marginal_mode = collect(marginalmode(sd))
        )
    return tab
end

function fixed_parameter_table(sd::SampledMeasure)
    fixed = fixedparams(sd)
    freekeys = collect(keys(fixed))

    tab = TypedTables.Table(
        parameter = freekeys,
        value = collect(fixed)
        )
    return tab
end


function Base.show(io::IO, mime::MIME"text/plain", sd::SampledMeasure)
    println(io, "BAT.jl - SampledMeasure")
    _line(io, length=30)

    println(io, "\nSampling:")
    _line(io, length=25)
    _print_generator(io, sd._generator)
    _print_sampling(io, sd)

    fpt = fixed_parameter_table(sd)
    if !isempty(fpt)
        println(io, "\n\nFixed parameters:")
        _line(io, length=25)
        println(io, "number of fixed parameters: ", nfixedparams(sd))
        println(io, "fixed parameters: ",fixedparams(sd))
    end

    println(io, "\n\nParameter estimates:")
    _line(io, length=25)
    println(io, "number of free parameters: ", nfreeparams(sd), "\n")
    println(io, parameter_table(sd))

    println(io, "\n\nCovariance matrix:")
    _line(io, length=25)
    println(io, cov(sd))
end


function _line(io::IO; length=25, indent=0)
    println(io, repeat(' ', indent), repeat('─', length))
end


function _print_sampling(io::IO, sd::SampledMeasure)
    println(io, "total number of samples:", repeat(' ', 6), numberofsamples(sd))
    println(io, "effective number of samples: ", eff_sample_size(sd))
end


function _print_generator(io::IO, generator::GenericSampleGenerator)
    print(io, "algorithm: ")
    show(io, "text/plain", getalgorithm(generator))
    print(io, "\n")
end

function _print_generator(io::IO, generator::UnknownSampleGenerator)
end

function _print_generator(io::IO, generator::MCMCSampleGenerator)
    chains = generator._chains
    nchains = length(chains)
    n_tuned_chains = count(c -> c.info.tuned, chains)
    n_converged_chains = count(c -> c.info.converged, chains)
    print(io, "algorithm: ")
    show(io, "text/plain", getalgorithm(generator))
    print(io, "\n")

    println(io, "number of chains:", repeat(' ', 13), nchains)
    println(io, "number of chains tuned:", repeat(' ', 7), n_tuned_chains)
    println(io, "number of chains converged:", repeat(' ', 3), n_converged_chains)
    println(io, "number of samples per chain:", repeat(' ', 2), nsamples(chains[1]), "\n")
end
