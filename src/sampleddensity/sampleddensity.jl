export SampledDensity


struct SampledDensity{D<:AbstractDensity,S<:DensitySampleVector, SI<:SamplerInfo}
    density::D
    samples::S
    stats::MCMCBasicStats
    samplerinfo::SI
end

function SampledDensity(
    density::AbstractPosteriorDensity,
    samples::DensitySampleVector;
    samplerinfo::SamplerInfo = NoSamplerInfo()
)
    stats = MCMCBasicStats(samples)
    return SampledDensity(density, samples, stats, samplerinfo)
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

import ValueShapes.varshape
function varshape(sd::SampledDensity)
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

import Statistics.mean
function mean(sd::SampledDensity)
    means = sd.stats.param_stats.mean
    return (; zip(active_keys(sd), means)...,)
end

import Statistics.std
function std(sd::SampledDensity)
    covm = collect(sd.stats.param_stats.cov)
    stds = sqrt.(LinearAlgebra.diag(covm))

    return (; zip(active_keys(sd), stds)...,)
end

function mode(sd::SampledDensity)
    modes = sd.stats.mode
    return (; zip(active_keys(sd), modes)...,)
end

function marginalmode(sd::SampledDensity)
    modes = unshaped(bat_marginalmode(sd.samples).result)
    return (; zip(active_keys(sd), modes)...,)
end

import Statistics.cov
function cov(sd::SampledDensity)
    covm = collect(sd.stats.param_stats.cov)
    names = string.(active_keys(sd))

    return NamedArrays.NamedArray(covm, (names, names), ("cov",""))
end

import Statistics.cor
function cor(sd::SampledDensity)
    covm = collect(sd.stats.param_stats.cov)
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
