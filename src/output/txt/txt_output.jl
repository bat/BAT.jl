function Base.show(io::IO, mime::MIME"text/plain", summary::Summary)
    println(io, "BAT.jl - Result Summary")
    _bold_line(io)
    _empty_line(io)

    println(io, "Model")
    _bold_line(io, length=6)
    _print_likelihood(io, mime, summary)
    _print_prior(io, mime, summary)

    _print_sampling(io, mime, summary) # algorithm, number of samples, (chains, convergence)

    # esults of sampling: mean, mode
    _print_parameter_results(io, summary)
    _print_covariance(io, summary)
end



function _bold_line(io::IO; length=25)
    println(io, repeat('=', length))
end

function _thin_line(io::IO; length=25, indent=0)
    println(io, repeat(' ', indent), repeat('-', length))
end

function _empty_line(io::IO; lines=1)
    println(io, repeat('\n', lines))
end



function _print_likelihood(io, m,  summary::Summary)
    likelihood = summary.posterior.likelihood

    print(io,  " likelihood:  ")
    _print_pretty(io, m, likelihood)

    if isa(likelihood, GenericDensity)
        file, line = functionloc(likelihood.f)
        println(io, "\n              defined in \"", file, "\", line ", line )
    end
end

function _print_prior(io, m,  summary::Summary)
    prior = summary.posterior.prior.dist._internal_distributions
    print(io, "\n prior:       ")

    for (i,k) in enumerate(keys(prior))
        i>1 ? print("              ") : nothing
        println(k, " = ", getindex(prior, k))
    end
    _empty_line(io)
end


function _print_sampling(io, m,  summary::Summary)
    println(io, "Sampling")
    _bold_line(io)

    print(io, " algorithm: ")
    _print_pretty(io, m, summary.samplerinfo.algorithm)
    _empty_line(io)

    println(io, " total number of samples:", repeat(' ', 6), summary.nsamples)
    _print_samplerinfo(io, m, summary.samplerinfo)

    _empty_line(io)
end


function _print_samplerinfo(io, m, samplerinfo::MCMCInfo)
    chains = samplerinfo.chains
    nchains = length(chains)
    n_tuned_chains = count(c -> c.info.tuned, chains)
    n_converged_chains = count(c -> c.info.converged, chains)

    println(io, " number of chains:", repeat(' ', 13), nchains)
    println(io, " number of chains tuned:", repeat(' ', 7), n_tuned_chains)
    println(io, " number of chains converged:", repeat(' ', 3), n_converged_chains)
    println(io, " number of samples per chain:", repeat(' ', 2), chains[1].nsamples)
end


function _print_samplerinfo(io, m, samplerinfo::ImportanceSamplerInfo)
end


function _print_parameter_results(io, summary)
    println(io, "Results")
    _bold_line(io)
    nparams = summary.stats.param_stats.m

    fixed_param_names = get_fixed_names(summary.shape)
    length(fixed_param_names) > 0 ? _print_fixed_parameters(io, samples, fixed_param_names) : nothing

    for p in 1:nparams
        pname = getstring(summary.shape, p)
        pstring = "$p. $pname:"
        println(io, pstring)
        _thin_line(io, length=length(pstring)+1)

        println(io, "  mean ± std.dev. = ", @sprintf("%.5g", summary.stats.param_stats.mean[p]),
                " ± ", @sprintf("%.5g", sqrt(summary.stats.param_stats.cov[p, p])))
        println(io, "  global mode     = ", @sprintf("%.5g",  summary.stats.mode[p]))
        println(io, "  marginal mode   = ",  @sprintf("%.5g", summary.marginalmodes[p]))
        print(io, "\n")
    end

end


function _print_fixed_parameters(io, samples, fixed_names)
    pstring = "fixed: "
    println(io, pstring)
    _thin_line(io, length=length(pstring)+1)
    for n in fixed_names
        println(io,  "  " * n*" = ", @sprintf("%.5g", getindex(samples[1].v, Symbol(n))))
        print(io, "\n")
    end
end


function _print_covariance(io, summary)
    println(io, "covariance matrix: ")
    _thin_line(io, length=19)
    cov = summary.stats.param_stats.cov
    Base.print_array(io, round.(cov, sigdigits=4))
end
