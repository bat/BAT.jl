function Base.show(io::IO, mime::MIME"text/plain", summary::Summary)
    stats = MCMCBasicStats(summary.samples)
    samples = summary.samples

    println(io, "BAT.jl - Result Summary")
    _bold_line(io)
    _empty_line(io)

    # print model: likelihood, prior
    _print_model(io, mime, summary)

    # print sampling: algorithm, number of samples, chains, convergence...
    _print_sampling(io, mime, summary)

    # print results of sampling: mean, mode
    println(io, "Results")
    _bold_line(io)
    _print_parameters(io, stats, samples)

    # print covariance
    _print_covariance(io, stats, samples)
end



function _bold_line(io::IO; length=25)
    println(io, repeat('=', length))
end

function _thin_line(io::IO; length=25, indent=0)
    println(io, repeat(' ', indent), repeat('-', length))
end

function _empty_line(io::IO; lines=1)
    print(io, repeat('\n', lines))
end



function _print_model(io, m,  summary::Summary)
    println(io, "Model")
    _bold_line(io, length=6)

    print(io,  " likelihood:  ")
    display_rich(io, m, summary.chains[1].spec.posterior.likelihood)

    if isa(summary.chains[1].spec.posterior.likelihood, GenericDensity)
        file, line = functionloc(summary.chains[1].spec.posterior.likelihood.f)
        println(io, "\n              defined in \"", file, "\", line ", line )
    end

    print(io, "\n prior:       ")
    a = summary.chains[1].spec.posterior.prior.dist._internal_distributions
    i = 0
    for k in keys(a)
        i>0 ? print("              ") : nothing
        println(k, " = ", getindex(a, k))
        i += 1
    end
    _empty_line(io)
end


function _print_sampling(io, m,  summary::Summary)
    println(io, "Sampling")
    _bold_line(io)

    algorithm = summary.chains[1].spec.algorithm

    print(io, " algorithm: ")
    display_rich(io, m, algorithm)
    _empty_line(io)

    nsamples = length(summary.samples)
    s = summary.chains[1].info.tuned

    println(io, " total number of samples:", repeat(' ', 6), nsamples)

    if _has_chains(algorithm)
        chains = summary.chains
        nchains = length(chains)
        n_tuned_chains = count(c -> c.info.tuned, chains)
        n_converged_chains = count(c -> c.info.converged, chains)

        println(io, " number of chains:", repeat(' ', 13), nchains)
        println(io, " number of chains tuned:", repeat(' ', 7), n_tuned_chains)
        println(io, " number of chains converged:", repeat(' ', 3), n_converged_chains)
        println(io, " number of samples per chain:", repeat(' ', 2), Int(nsamples/nchains))
    end
    _empty_line(io)
end


function _has_chains(algorithm::AbstractSamplingAlgorithm)
    isa(algorithm, ImportanceSampler) ? (return false) : (return true)
end


#TODO
function _print_parameters(io, stats, samples)
    nparams = stats.param_stats.m

    for p in 1:nparams
        pname = getstring(samples, p)
        pstring = "$p. $pname:"
        println(io, pstring)
        _thin_line(io, length=length(pstring)+1)

        println(io, "  mean ± std.dev. = ", @sprintf("%.5g",stats.param_stats.mean[p]),
                " ± ", @sprintf("%.5g", sqrt(stats.param_stats.cov[p, p])))
        println(io, "  global mode     = ", @sprintf("%.5g",stats.mode[p]))
        println(io, "  marginal mode   = ", "not yet implemented (WIP)")
        print(io, "\n")
    end

    fixed_names = get_fixed_names(varshape(samples))
    if length(fixed_names) > 0
        _print_fixed_parameters(io, samples, fixed_names)
    end

end

function _print_fixed_parameters(io, samples, fixed_names)
    pstring = "fixed parameters: "
    println(io, pstring)
    _thin_line(io, length=length(pstring)+1)
    for n in fixed_names
        println(io,  "  " * n*" = ", @sprintf("%.5g", getindex(samples[1].v, Symbol(n))))
        print(io, "\n")
    end
end



function _print_covariance(io, stats, samples)
    println(io, "covariance matrix: ")
    _thin_line(io, length=19)
    cov = stats.param_stats.cov
    Base.print_array(io, round.(cov, sigdigits=6))

    # nparams = stats.param_stats.m
    # for i in 1:nparams
    #     print(io, "  ")
    #     for j in 1:nparams
    #         s = @sprintf("%.5g", stats.param_stats.cov[i, j])
    #         print(io, s, "  ")
    #     end
    #     print(io, "\n")
    # end
end





function Base.show(io::IO, m::MIME"text/plain", algorithm::MetropolisHastings)
    proposal = algorithm.proposalspec
    weighting = algorithm.weighting
    println(io, "MetropolisHastings(", proposal, ", ", weighting, ")")
end
