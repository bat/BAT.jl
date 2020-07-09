# # BAT.jl plotting tutorial

using BAT
using Distributions
using IntervalSets
using ValueShapes

# ## Generate samples to be plotted
likelihood = params -> begin

    r1 = logpdf.(
    MixtureModel(Normal[
    Normal(-10.0, 1.2),
    Normal(0.0, 1.8),
    Normal(10.0, 2.5)], [0.1, 0.3, 0.6]), params.a)

    r2 = logpdf.(
    MixtureModel(Normal[
    Normal(-5.0, 2.2),
    Normal(5.0, 1.5)], [0.3, 0.7]), params.c[1])

    r3 = logpdf.(Normal(2.0, 1.5), params.c[2])

    return LogDVal(r1+r2+r3)
end

prior = BAT.NamedTupleDist(
    a = Normal(-3, 4.5),
    #b = 0,
    c = [-20.0..20.0, -10..10]
)

posterior = PosteriorDensity(likelihood, prior);

samples, summary, chains = bat_sample(posterior, 10^5, SobolSampler());
display(summary)


#nb # ### plain text output (for terminal)
#nb show(stdout,"text/plain", summary)

# ### write output to txt-file
io = open("output.txt", "w")
show(io, "text/plain", summary)
close(io)


# ### write output to html-file
io = open("output.html", "w")
show(io, "text/html", summary)
close(io)
