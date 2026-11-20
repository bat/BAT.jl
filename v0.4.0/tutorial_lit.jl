#md # # Tutorial
#nb # # BAT.jl Tutorial
#jl # # BAT.jl Tutorial
#
# This tutorial demonstrates a simple application of BAT.jl: A Bayesian fit
# of a histogram with two Gaussian peaks.
#
#md # You can also download this tutorial as a
#md # [Jupyter notebook](bat_tutorial.ipynb) and a plain
#md # [Julia source file](bat_tutorial.jl).
#
#md # Table of contents:
#md #
#md # ```@contents
#md # Pages = ["tutorial.md"]
#md # Depth = 3
#md # ```
#
# Note: This tutorial is somewhat verbose, as it aims to be easy to follow for
# users who are new to Julia. For the same reason, we deliberately avoid making
# use of Julia features like
# [closures](https://docs.julialang.org/en/v1/devdocs/functions/#Closures-1),
# [anonymous functions](https://docs.julialang.org/en/v1/manual/functions/index.html#man-anonymous-functions-1),
# [broadcasting syntax](https://docs.julialang.org/en/v1/manual/arrays/index.html#Broadcasting-1),
# [performance annotations](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-annotations-1),
# etc.


# ## Input Data Generation
#
# First, let's generate some synthetic data to fit. We'll need the Julia
# standard-library packages
# ["Random"](https://docs.julialang.org/en/v1/stdlib/Random/),
# ["LinearAlgebra"](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
# and ["Statistics"](https://docs.julialang.org/en/v1/stdlib/Statistics/),
# as well as the packages
# ["Distributions"](https://juliastats.github.io/Distributions.jl/stable/)
# and ["StatsBase"](http://juliastats.github.io/StatsBase.jl/stable/):

using Random, LinearAlgebra, Statistics, Distributions, StatsBase

# As the underlying truth of our input data/histogram, let us choose an
# non-normalized probability density composed of two Gaussian peaks with a peak
# area of 500 and 1000, a mean of -1.0 and 2.0 and a standard error of 0.5

data = vcat(
    rand(Normal(-1.0, 0.5), 500),
    rand(Normal( 2.0, 0.5), 1000)
)

# resulting in a vector of floating-point numbers:

typeof(data) == Vector{Float64}

# Next, we'll create a histogram of that data, this histogram will serve as
# the input for the Bayesian fit:

hist = append!(Histogram(-2:0.1:4), data)


# Using the Julia ["Plots"](http://docs.juliaplots.org/latest/) package

using Plots

# we can plot the histogram:

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data"
)
#jl savefig("tutorial-data.pdf")
#md savefig("tutorial-data.pdf")
#md savefig("tutorial-data.svg"); nothing # hide
#md # [![Data](tutorial-data.svg)](tutorial-data.pdf)


# The package ["EponymTuples"](https://github.com/tpapp/EponymTuples.jl)
# provides a very useful macro `@eponymargs`: It makes it easy to define
# functions that take named tuples as arguments and unpack them.

using EponymTuples

# This comes in handy for the definition of ou fit function - the function
# that we expect to describes the data histogram (depending on some model
# parameters):

function fit_function(@eponymargs(a, mu, sigma), x::Real)
    a[1] * pdf(Normal(mu[1], sigma), x) +
    a[2] * pdf(Normal(mu[2], sigma), x)
end
#md nothing # hide

# The fit parameters (model parameters) `a` (peak areas) and `mu` (peak means)
# are vectors, parameter `sigma` (peak width) is a scalar, we assume it's the
# same for both Gaussian peaks.
#
# The true values for the model/fit parameters are:

true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)
#md nothing # hide

# Let's visually compare the histogram and the fit function, using the true
# parameter values, to make sure everything is set up correctly:

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data and True Statistical Model"
)
plot!(
    -4:0.01:4, x -> fit_function(true_par_values, x),
    label = "Truth"
)
#jl savefig("tutorial-data-and-truth.pdf")
#md savefig("tutorial-data-and-truth.pdf")
#md savefig("tutorial-data-and-truth.svg"); nothing # hide
#md # [![Data and True Statistical Model](tutorial-data-and-truth.svg)](tutorial-data-and-truth.pdf)


# ## Bayesian Fit
#
# Now we'll perform a Bayesian fit of the generated histogram, using BAT,
# to infer the model parameters from the data histogram.
#
# In addition to the Julia packages loaded above, we need BAT itself, as
# well as [IntervalSets](https://github.com/JuliaMath/IntervalSets.jl):

using BAT, IntervalSets


# ### Likelihood Definition
#
# First, we need to define a likelihood function for our problem. In BAT,
# all likelihood functions and priors are subtypes of `BAT.AbstractDensity`. 
# We'll store the histogram that we want to fit in our likelihood density
# type, as accessing the histogram as a global variable would
# [reduce performance](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Avoid-global-variables-1):

struct HistogramLikelihood{H<:Histogram,F<:Function} <: AbstractDensity
    histogram::H
    fitfunc::F
end

# As a minimum, BAT requires a method `BAT.density_logval`
# to be defined for each subtype of `AbstractDensity`.
#
# `BAT.density_logval` implements the actual log-likelihood function:

function BAT.density_logval(
    likelihood::HistogramLikelihood,
    params::Union{NamedTuple,AbstractVector{<:Real}}
)
    ## Histogram counts for each bin as an array:
    counts = likelihood.histogram.weights

    ## Histogram binning, has length (length(counts) + 1):
    binning = likelihood.histogram.edges[1]

    ## sum log-likelihood over bins:
    log_likelihood::Float64 = 0.0
    for i in eachindex(counts)
        bin_left, bin_right = binning[i], binning[i+1]
        bin_width = bin_right - bin_left
        bin_center = (bin_right + bin_left) / 2

        observed_counts = counts[i]

        ## Simple mid-point rule integration of fitfunc over bin:
        expected_counts = bin_width * likelihood.fitfunc(params, bin_center)

        log_likelihood += logpdf(Poisson(expected_counts), observed_counts)
    end

    return log_likelihood
end


# BAT makes use of Julia's parallel programming facilities if possible, e.g.
# to run multiple Markov chains in parallel, and expects implementations of
# `BAT.density_logval` to be thread safe. Mark non-thread-safe code with
# `@critical` (using Julia package `ParallelProcessingTools`).
#
# BAT requires Julia v1.3 or newer to use multi-threading. Support for
# automatic parallelization across multiple (local and remote) Julia processes
# is planned, but not implemented yet.
#
# Note that Julia currently starts only a single thread by default, you will
# need to set the environment variable
# [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1)
# to specify the number of Julia threads.
#
# Using our likelihood density definition and the histogram to fit, we can now
# create our data- and fit-function-specific likelihood instance:

likelihood = HistogramLikelihood(hist, fit_function)


# ### Prior Definition
#
# Next, we need to choose a sensible prior for the fit:

prior = NamedPrior(
    a = [0.0..10.0^4, 0.0..10.0^4],
    mu = [-2.0..0.0, 1.0..3.0],
    sigma = Truncated(Normal(0.4, 2), 0.3, 0.7)
)
#md nothing # hide

# In general, BAT allows instances of any subtype of `AbstractDensity` to
# be uses as a prior, as long as a sampler is defined for it. This way, users
# may implement complex application-specific priors. You can also
# use `convert(AbstractDensity, distribution)` to convert any
# continuous multivariate `Distributions.Distribution` to a
# `BAT.AbstractDensity` that can be used as a prior (or likelihood).
#
# The prior also implies the shapes of the parameters:

using ValueShapes

parshapes = valshape(prior)

# These will come in handy later on, e.g. to access (the posterior
# distribution of) individual parameter values.


# ### Bayesian Model Definition
#
# Given the likelihood and prior definition, a `BAT.PosteriorDensity` is simply
# defined via

posterior = PosteriorDensity(likelihood, prior)
#md nothing # hide


# ### Parameter Space Exploration via MCMC
#
# We can now use Markov chain Monte Carlo (MCMC) to explore the space of
# possible parameter values for the histogram fit.
#
# We'll use the Metropolis-Hastings algorithm and a multivariate
# t-distribution (ν = 1) as it's proposal function:

algorithm = MetropolisHastings(MvTDistProposalSpec(1.0))
#md nothing # hide

# We also need to which random number generator and seed to use. BAT requires
# a counter-based RNG and partitions the RNG space over the MCMC chains. This
# way, a single RNG seed is sufficient for all chains and results can be
# reproducible even under parallel execution. Let's choose a Philox4x RNG
# with a random seed:

rngseed = BAT.Philox4xSeed()
#md nothing # hide

# The algorithm, posterior and RNG seed specify the MCMC chains:

chainspec = MCMCSpec(algorithm, posterior, rngseed)
#md nothing # hide

# Let's use 4 MCMC chains and require 10^5 unique samples from each chain
# (after tuning/burn-in):

nsamples = 10^4
nchains = 4
#md nothing # hide

# BAT provides fine-grained control over the MCMC tuning algorithm,
# convergence test and the chain initialization and tuning/burn-in strategy
# (the values used here are the default values):

tuner_config = ProposalCovTunerConfig(
    λ = 0.5,
    α = 0.15..0.35,
    β = 1.5,
    c = 1e-4..1e2
)

convergence_test = BGConvergence(
    threshold = 1.1,
    corrected = false
)

init_strategy = MCMCInitStrategy(
    ninit_tries_per_chain = 8..128,
    max_nsamples_pretune = 25,
    max_nsteps_pretune = 250,
    max_time_pretune = Inf
)

burnin_strategy = MCMCBurninStrategy(
    max_nsamples_per_cycle = 1000,
    max_nsteps_per_cycle = 10000,
    max_time_per_cycle = Inf,
    max_ncycles = 30
)

#md nothing # hide

# To increase the verbosity level of BAT logging output, you may want to set
# the Julia logging level for BAT to debug via `ENV["JULIA_DEBUG"] = "BAT"`.

#nb ENV["JULIA_DEBUG"] = "BAT"
#jl ENV["JULIA_DEBUG"] = "BAT"

# Now we can generate a set of MCMC samples via `rand`:

samples, sampleids, stats, chains = rand(
    chainspec,
    nsamples,
    nchains,
    tuner_config = tuner_config,
    convergence_test = convergence_test,
    init_strategy = init_strategy,
    burnin_strategy = burnin_strategy,
    max_nsteps = 10^5,
    max_time = Inf,
    granularity = 1
)
#md nothing # hide

# Note: Reasonable default values are defined for all of the above. In many
# use cases, a simple
#
# ```julia
# samples, sampleids, stats, chains =
#    rand(MCMCSpec(MetropolisHastings(), model), nsamples, nchains)`
# ```
#
# may be sufficient.

# Let's print some results:

println("Truth: $true_par_values")
println("Mode: $(stats.mode)")
println("Mean: $(stats.param_stats.mean)")
println("Covariance: $(stats.param_stats.cov)")

# `stats` contains some statistics collected during MCMC sample generation,
# e.g. the mean and covariance of the parameters and the mode. Equal values
# for these statistics may of course be calculated afterwards, from the
# samples:

@assert vec(mean(samples.params, FrequencyWeights(samples.weight))) ≈ stats.param_stats.mean
@assert vec(var(samples.params, FrequencyWeights(samples.weight))) ≈ diag(stats.param_stats.cov)
@assert cov(samples.params, FrequencyWeights(samples.weight)) ≈ stats.param_stats.cov

# We can also, e.g., get the Pearson auto-correlation of the parameters:

vec(cor(samples.params, FrequencyWeights(samples.weight)))


# ### Visualization of Results

# BAT.jl comes with an extensive set of plotting recipes for
# ["Plots.jl"] (http://docs.juliaplots.org/latest/).
# We can plot the marginalized distribution for a single parameter (e.g.
# parameter 3, i.e. μ₁):

par_names = ["a_1", "a_2", "mu_1", "mu_2", "sigma"]
plot(
    samples, 3,
    mean = true, std_dev = true, globalmode = true, localmode = true,
    nbins = 50, xlabel = par_names[3], ylabel = "P($(par_names[3]))",
    title = "Marginalized Distribution for mu_1"
)
#jl savefig("tutorial-single-par.pdf")
#md savefig("tutorial-single-par.pdf")
#md savefig("tutorial-single-par.svg"); nothing # hide
#md # [![Marginalized Distribution for mu_1](tutorial-single-par.svg)](tutorial-single-par.pdf)

# or plot the marginalized distribution for a pair of parameters (e.g.
# parameters 3 and 5, i.e. μ₁ and σ), including information from the parameter
# stats:

plot(
    samples, (3, 5),
    mean = true, std_dev = true, globalmode = true, localmode = true,
    nbins = 50, xlabel = par_names[3], ylabel = par_names[5],
    title = "Marginalized Distribution for mu_1 and sigma"
)
plot!(stats, (3, 5))
#jl savefig("tutorial-param-pair.pdf")
#md savefig("tutorial-param-pair.pdf")
#md savefig("tutorial-param-pair.svg"); nothing # hide
#md # [![Marginalized Distribution for mu_1 and sigma](tutorial-param-pair.svg)](tutorial-param-pair.pdf)

# We can also create an overview plot of the marginalized distribution for all
# pairs of parameters:

plot(
    samples,
    mean = false, std_dev = false, globalmode = true, localmode = false,
    nbins = 50
)
#jl savefig("tutorial-all-params.pdf")
#md savefig("tutorial-all-params.pdf")
#md savefig("tutorial-all-params.svg"); nothing # hide
#md # [![Pairwise Correlation between Parameters](tutorial-all-params.svg)](tutorial-all-params.pdf)


# ### Integration with Tables.jl

# BAT.jl supports the [Tables.jl](https://github.com/JuliaData/Tables.jl)
# interface. Using a tables implementation like
# TypedTables.jl](http://blog.roames.com/TypedTables.jl/stable/),
# the whole MCMC output (parameter vectors, weights, sample/chain numbers,
# etc.) can easily can be combined into a single table:

using TypedTables

tbl = Table(samples, sampleids)


# Using the parameter shapes, we can also generate a table with named
# parameters instead:

tbl_named = Table(parshapes.(samples), sampleids)


# We can now, e.g., find the sample with the maximum posterior value (i.e. the
# mode):

mode_log_posterior, mode_idx = findmax(tbl_named.log_posterior)

# And get row `mode_idx` of the table, with all information about the sample
# at the mode:


# ## Comparison of Truth and Best Fit

# As a final step, we retrieve the parameter values at the mode, representing
# the best-fit parameters

fit_par_values = tbl_named[mode_idx].params

# And plot the truth, data, and best fit:

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data, True Model and Best Fit"
)
plot!(-4:0.01:4, x -> fit_function(true_par_values, x), label = "Truth")
plot!(-4:0.01:4, x -> fit_function(fit_par_values, x), label = "Best fit")
#jl savefig("tutorial-data-truth-bestfit.pdf")
#md savefig("tutorial-data-truth-bestfit.pdf")
#md savefig("tutorial-data-truth-bestfit.svg"); nothing # hide
#md # [![Data, True Model and Best Fit](tutorial-data-truth-bestfit.svg)](tutorial-data-truth-bestfit.pdf)
