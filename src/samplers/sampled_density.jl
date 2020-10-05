export SampledDensity

struct SampledDensity{D<:AbstractDensity,S<:DensitySampleVector, G<:AbstractSampleGenerator}
    density::D
    samples::S
    _stats::MCMCBasicStats
    generator::G
end

function SampledDensity(
    density::AbstractPosteriorDensity,
    samples::DensitySampleVector;
    generator::AbstractSampleGenerator = UnknownSampleGenerator()
)
    stats = MCMCBasicStats(samples)
    return SampledDensity(density, samples, stats, generator)
end



function density(sd::SampledDensity)
    return sd.density
end

function nfreeparams(sd::SampledDensity)
    return varshape(sd)._flatdof
end

function freeparams(sd::SampledDensity)
    return active_keys(sd)
end

function nfixedparams(sd::SampledDensity)
    return length(get_fixed_names(varshape(sd)))
end

function ValueShapes.varshape(sd::SampledDensity)
    return varshape(sd.samples)
end

function fixedparams(sd::SampledDensity)
    param_shape = varshape(sd)
    fixed_param_keys = Symbol.(get_fixed_names(param_shape))
    fixed_values = [getproperty(param_shape, f).shape.value for f in fixed_param_keys]
    return (; zip(fixed_param_keys, fixed_values)...,)
end

function numberofsamples(sd::SampledDensity)
    return length(sd.samples)
end

function eff_sample_size(sd::SampledDensity)
    effsize = bat_eff_sample_size(unshaped.(sd.samples)).result
    return (; zip(active_keys(sd), effsize)...,)

end

function active_keys(sd::SampledDensity)
    return Symbol.(all_active_names(sd.samples.v.__internal_elshape))
end

function Statistics.mean(sd::SampledDensity)
    means = sd._stats.param_stats.mean
    return (; zip(active_keys(sd), means)...,)
end

function Statistics.std(sd::SampledDensity)
    covm = collect(sd._stats.param_stats.cov)
    stds = sqrt.(LinearAlgebra.diag(covm))

    return (; zip(active_keys(sd), stds)...,)
end

function Distributions.mode(sd::SampledDensity)
    modes = sd._stats.mode
    return (; zip(active_keys(sd), modes)...,)
end

function marginalmode(sd::SampledDensity)
    modes = unshaped(bat_marginalmode(sd.samples).result)
    return (; zip(active_keys(sd), modes)...,)
end

function Statistics.cov(sd::SampledDensity)
    covm = collect(sd._stats.param_stats.cov)
    names = string.(active_keys(sd))

    return NamedArrays.NamedArray(covm, (names, names), ("cov",""))
end

function Statistics.cor(sd::SampledDensity)
    covm = collect(sd._stats.param_stats.cov)
    corm = cov2cor(covm, sqrt.(diag(covm)))
    names = string.(active_keys(sd))

    return NamedArrays.NamedArray(corm, (names, names), ("cor",""))
end


function parameter_table(sd::SampledDensity)
    tab = TypedTables.Table(
        parameter = active_keys(sd),
        mean = collect(mean(sd)),
        std = collect(std(sd)),
        global_mode = collect(mode(sd)),
        marginal_mode = collect(marginalmode(sd))
        )
    return tab
end

function fixed_parameter_table(sd::SampledDensity)
    fixed = fixedparams(sd)
    freekeys = collect(keys(fixed))

    tab = TypedTables.Table(
        parameter = freekeys,
        value = collect(fixed)
        )
    return tab
end


function Base.show(io::IO, mime::MIME"text/plain", sd::SampledDensity)
    println(io, "BAT.jl - SampledDensity")
    _line(io, length=30)

    println(io, "\nSampling:")
    _line(io, length=25)
    _print_generator(io, sd.generator)
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


function _print_sampling(io::IO, sd::SampledDensity)
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
    println(io, "number of samples per chain:", repeat(' ', 2), chains[1].nsamples, "\n")
end


get_initsrc_from_target(target::SampledDensity) = target.samples
