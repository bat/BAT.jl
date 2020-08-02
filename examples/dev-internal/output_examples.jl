using BAT
using Distributions
using IntervalSets
using TypedTables
using NamedArrays
using LinearAlgebra
using PrettyTables


likelihood = params -> begin
    r1 = logpdf.(Normal(-10.0, 1.2), params.a)
    r2 = logpdf.(Normal(-5.0, 0.8), params.c[1])
    r3 = logpdf.(Normal(12.0, 1.4), params.c[2])

    return LogDVal(r1+r2+r3)
end

prior = BAT.NamedTupleDist(
    a = Normal(-3, 4.5),
    b = 0,
    c = [-20.0..20.0, -10..10]
)

posterior = PosteriorDensity(likelihood, prior);

samples, chains = bat_sample(posterior, (10^5, 4), MetropolisHastings());
#samples, summary, chains = bat_sample(posterior, 10^5, SobolSampler());

sd = SampledDensity(posterior, samples, samplerinfo=BAT.MCMCSamplerInfo(MetropolisHastings(), chains))
display(sd)



# # write summary to txt-file
# io = open("summary.txt", "w")
# show(io, "text/plain", summary)
# close(io)
#
# # ### write output to html-file
# io = open("output.html", "w")
# show(io, "text/html", summary)
# close(io)
