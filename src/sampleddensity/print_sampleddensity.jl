function Base.show(io::IO, mime::MIME"text/plain", sd::SampledDensity)
    println(io, "BAT.jl - SampledDensity")
    _line(io, length=30)

    println(io, "\nSampling:")
    _line(io, length=25)
    _print_samplerinfo(io, sd.samplerinfo)
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


function _print_samplerinfo(io::IO, samplerinfo::GenericSamplerInfo)
    print("algorithm: ")
    show(io, "text/plain", samplerinfo.algorithm)
    print("\n")
end

function _print_samplerinfo(io::IO, samplerinfo::NoSamplerInfo)
end

function _print_samplerinfo(io::IO, samplerinfo::MCMCSamplerInfo)
    chains = samplerinfo.chains
    nchains = length(chains)
    n_tuned_chains = count(c -> c.info.tuned, chains)
    n_converged_chains = count(c -> c.info.converged, chains)
    print("algorithm: ")
    show(io, "text/plain", samplerinfo.algorithm)
    print("\n")

    println(io, "number of chains:", repeat(' ', 13), nchains)
    println(io, "number of chains tuned:", repeat(' ', 7), n_tuned_chains)
    println(io, "number of chains converged:", repeat(' ', 3), n_converged_chains)
    println(io, "number of samples per chain:", repeat(' ', 2), chains[1].nsamples, "\n")
end
