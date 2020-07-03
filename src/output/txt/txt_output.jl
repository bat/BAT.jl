
function Base.show(io::IO, m::MIME"text/plain", prior::ConstDensity{<:HyperRectBounds}; tabs=false)
    println(io, "HyperRectBounds")

    for p in 1:length(prior.bounds.vol.hi)
        print(io, "\t \t $p.")
        print(io, prior.bounds.bt[p])
        println(io, " [", prior.bounds.vol.lo[p],", ", prior.bounds.vol.hi[p],"]")
    end

end


function Base.show(io::IO, m::MIME"text/plain", algorithm::MetropolisHastings)
    proposal = algorithm.proposalspec
    weighting = algorithm.weighting
    println(io, "MetropolisHastings(", proposal, ", ", weighting, ")")
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

function Base.show(io::IO, m::MIME"text/plain", summary::Summary)

    stats = MCMCBasicStats(summary.samples)

    println(io, "BAT.jl - Result Summary")
    _bold_line(io)
    _empty_line(io)


    println(io, "Model")
    println(io, repeat('=', 6))

    # print(io,  "  likelihood:  ")
    # display_rich(io, m, summary.chains[1].spec.model.likelihood)
    # print(io, "\n")
    #
    # print(io, "  prior:       ")
    # display_rich(io, m, summary.chains[1].spec.model.prior)
    # print(io, "\n")
    # print(io, "\n")

    _print_sampling(io, m, summary)

    println(io, "Results")
    _bold_line(io)
    _print_parameters(io, stats, summary.samples)

    println(io, "covariance matrix: ")
    _thin_line(io, length=19)
    _print_covariance(io, stats, summary.samples)

    println(io, "\n")
end

function _print_sampling(io, m,  summary::Summary)
    println(io, "Sampling")
    _bold_line(io)

    print(io, "  algorithm: ")
    display_rich(io, m, summary.chains[1].spec.algorithm)

    nchains = length(summary.chains)
    nsamples = length(summary.samples)

    println(io, "  number of chains:", repeat(' ', 13), nchains)
    println(io, "  number of samples per chain:", repeat(' ', 2), Int(nsamples/nchains))
    println(io, "  total number of samples:", repeat(' ', 6), nsamples)
    _empty_line(io)
end


function _print_covariance(io, stats, samples)
    nparams = stats.param_stats.m
    for i in 1:nparams
        print(io, "  ")
        for j in 1:nparams
            s = @sprintf("%.5g", stats.param_stats.cov[i, j])
            print(io, s, "  ")
        end
        print(io, "\n")
     end
end


#TODO
function _print_parameters(io, stats, samples)
    nparams = stats.param_stats.m
    vs = varshape(samples)

    active_names = all_active_names(vs)
    all_names = allnames(vs)
    fixed_names = [n for n in all_names if !in(n, active_names)]


    for p in 1:nparams
        pname = getstring(samples, p)
        pstring = "$p. $pname:"
        println(io, pstring)
        _thin_line(io, length=length(pstring)+1)

        println(io, "  mean ± std.dev. = ", @sprintf("%.5g",stats.param_stats.mean[p]),
                " ± ", @sprintf("%.5g", sqrt(stats.param_stats.cov[p, p])))
        println(io, "  global mode     = ", @sprintf("%.5g",stats.mode[p]))
        println(io, "  marginal mode   = ", "not yet implemented")
        print(io, "\n")
    end


    pstring = "fixed: "
    println(io, pstring)
    _thin_line(io, length=length(pstring)+1)
    for n in fixed_names
        println(io,  "  " * n*" = ", @sprintf("%.5g", getindex(samples[1].v, Symbol(n))))
        print(io, "\n")
    end
end


# function Base.show(io::IO, m::MIME"text/plain", summary::Summary)
#
#     stats = summary.stats
#     nparams = length(stats.param_stats.mean)
#
#     println(io, "BAT.jl - Summary")
#     println(io, repeat('=', 18),"\n\n")
#
#
#     println(io, "Model")
#     println(io, repeat('=', 6))
#
#     print(io,  "  likelihood:  ")
#     display_rich(io, m, summary.chainresults[1].spec.model.likelihood)
#     print(io, "\n")
#
#     print(io, "  prior:       ")
#     display_rich(io, m, summary.chainresults[1].spec.model.prior)
#     print(io, "\n")
#     print(io, "\n")
#
#
#     println(io, "Sampling")
#     println(io, repeat('=', 9))
#
#     print(io,"  algorithm:", repeat(' ', 16))
#     display_rich(io, m, summary.chainresults[1].spec.algorithm)
#     print(io, "\n")
#
#     println(io, "  number of chains:",repeat(' ', 9), length(summary.chainresults))
#     println(io, "  total number of samples:",repeat(' ', 2), stats.param_stats.cov.n)
#     println(io, "\n")
#
#
#     println(io, "  Results")
#     println(io, "  ", repeat('=', 8))
#
#     for p in 1:nparams
#         println(io, "    parameter $p:")
#         println(io, "       mean ± std.dev. = ", @sprintf("%.3f",stats.param_stats.mean[p]),
#                 " ± ", @sprintf("%.3f", sqrt(stats.param_stats.cov[p, p])))
#         println(io, "       global mode     = ", @sprintf("%.3f",stats.mode[p]))
#         print(io, "\n")
#     end
#
#     println(io, "\n    covariance matrix: ")
#     for i in 1:nparams
#         print(io, "\t")
#         for j in 1:nparams
#             s = @sprintf("%.3f", stats.param_stats.cov[i, j])
#             print(io, s, "  ")
#         end
#         print(io, "\n")
#      end
#
# end
