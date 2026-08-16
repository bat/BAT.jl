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


# Let's define our fit function - the function that we expect to describe the
# data histogram, at each x-Axis position `x`, depending on a given set `p` of
# model parameters:

function fit_function(p::NamedTuple{(:a, :mu, :sigma)}, x::Real)
    p.a[1] * pdf(Normal(p.mu[1], p.sigma), x) +
    p.a[2] * pdf(Normal(p.mu[2], p.sigma), x)
end
#md nothing # hide

# The fit parameters (model parameters) `a` (peak areas) and `mu` (peak means)
# are vectors, parameter `sigma` (peak width) is a scalar, we assume it's the
# same for both Gaussian peaks.
#
# The true values for the model/fit parameters are the values we used to
# generate the data:

true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)
#md nothing # hide

# Let's visually compare the histogram and the fit function, using these true
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
# First, we need to define the likelihood (function) for our problem.
#
# BAT represents densities like likelihoods and priors as subtypes of
# `BAT.AbstractDensity`. Custom likelihood can be defined by
# creating a new subtype of `AbstractDensity` and by implementing (at minimum)
# `BAT.density_logval` for that type - in complex uses cases, this may become
# necessary. Typically, however, it is sufficient to define a custom
# likelihood as a simple function that returns the log-likelihood value for
# a given set of parameters. BAT will automatically convert such a
# log-likelihood function into a subtype of `AbstractDensity`.
#
# For performance reasons, functions should [not access global variables
# directly] (https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Avoid-global-variables-1).
# So we'll use an [anonymous function](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions-1)
# inside of a [let-statement](https://docs.julialang.org/en/v1/base/base/#let)
# to capture the value of the global variable `hist` in a local variable `h`
# (and to shorten function name `fit_function` to `f`, purely for
# convenience):

log_likelihood = let h = hist, f = fit_function
    params -> begin
        ## Histogram counts for each bin as an array:
        counts = h.weights

        ## Histogram binning, has length (length(counts) + 1):
        binning = h.edges[1]

        ## sum log-likelihood value over bins:
        ll_value::Float64 = 0.0
        for i in eachindex(counts)
            ## Get information about current bin:
            bin_left, bin_right = binning[i], binning[i+1]
            bin_width = bin_right - bin_left
            bin_center = (bin_right + bin_left) / 2

            observed_counts = counts[i]

            ## Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_width * f(params, bin_center)

            ## Add log of Poisson probability for bin:
            ll_value += logpdf(Poisson(expected_counts), observed_counts)
        end

        return ll_value
    end
end

# BAT makes use of Julia's parallel programming facilities if possible, e.g.
# to run multiple Markov chains in parallel. Therefore, log-likelihood
# (and other) code must be thread-safe. Mark non-thread-safe code with
# `@critical` (provided by Julia package `ParallelProcessingTools`).
#
# BAT requires Julia v1.3 or newer to use multi-threading. Support for
# automatic parallelization across multiple (local and remote) Julia processes
# is planned, but not implemented yet.
#
# Note that Julia currently starts only a single thread by default. Set the
# the environment variable
# [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1)
# to specify the desired number of Julia threads.

# We can evaluate `log_likelihood`, e.g. for the true parameter values:

log_likelihood(true_par_values)


# ### Prior Definition
#
# Next, we need to choose a sensible prior for the fit:

prior = NamedTupleDist(
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

posterior = PosteriorDensity(log_likelihood, prior)
#md nothing # hide


# ### Parameter Space Exploration via MCMC
#
# We can now use Markov chain Monte Carlo (MCMC) to explore the space of
# possible parameter values for the histogram fit.
#
# To increase the verbosity level of BAT logging output, you may want to set
# the Julia logging level for BAT to debug via `ENV["JULIA_DEBUG"] = "BAT"`.

#nb ENV["JULIA_DEBUG"] = "BAT"
#jl ENV["JULIA_DEBUG"] = "BAT"

# Let's use 4 MCMC chains and require 10^5 unique samples from each chain
# (after tuning/burn-in):

nsamples = 10^4
nchains = 4
#md nothing # hide


# Now we can generate a set of MCMC samples via [`bat_sample`](@ref):

samples, stats = bat_sample(posterior, (nsamples, nchains), MetropolisHastings())
#md nothing # hide
#nb nothing # hide

# Let's calculate some posterior statistics using the function
# [`bat_stats`](@ref) and print the results:

println("Truth: $true_par_values")
println("Mode: $(stats.mode)")
println("Mean: $(stats.mean)")
println("Covariance: $(stats.cov)")

# We can also, e.g., get the Pearson auto-correlation of the parameters:

cor(samples.params, FrequencyWeights(samples.weight))


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
plot!(MCMCBasicStats(samples), (3, 5))
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
# interface. So we can also convert the vector of MCMC samples vecto a
# table, e.g. using TypedTables.jl](http://blog.roames.com/TypedTables.jl/stable/):

using TypedTables

tbl = Table(samples)


# Using the parameter shapes, we can generate a table with named parameters,
# instead of flat real-valued parameter vectors:

tbl_named = parshapes.(samples)


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


# ## Fine-grained control
#
# BAT provides fine-grained control over the MCMC algorithm options, the
# MCMC chain initialization, tuning/burn-in strategy and convergence testing.
# All option value used in the following are the default values, any or all
# may be omitted.

# We'll sample using the The Metropolis-Hastings MCMC algorithm. By default,
# BAT uses a multivariate t-distribution (ν = 1) as the proposal function:

algorithm = MetropolisHastings(MvTDistProposal(1.0))
#md nothing # hide

# BAT requires a counter-based random number generator (RNG), since it
# partitions the RNG space over the MCMC chains. This way, a single RNG seed
# is sufficient for all chains and results are reproducible even under
# parallel execution. By default, BAT uses a Philox4x RNG initialized with a
# random seed drawn from the
# [system entropy pool](https://docs.julialang.org/en/v1/stdlib/Random/index.html#Random.RandomDevice):

using Random123
rng = Philox4x()
#md nothing # hide


# Other default parameters are:

tuning = AdaptiveMetropolisTuning(
    λ = 0.5,
    α = 0.15..0.35,
    β = 1.5,
    c = 1e-4..1e2
)

convergence = BrooksGelmanConvergence(
    threshold = 1.1,
    corrected = false
)

init = MCMCInitStrategy(
    ninit_tries_per_chain = 8..128,
    max_nsamples_pretune = 25,
    max_nsteps_pretune = 250,
    max_time_pretune = Inf
)

burnin = MCMCBurninStrategy(
    max_nsamples_per_cycle = 1000,
    max_nsteps_per_cycle = 10000,
    max_time_per_cycle = Inf,
    max_ncycles = 30
)

#md nothing # hide

# To generate MCMC samples with explicit control over all options, use

samples, stats = bat_sample(
    rng, posterior, (nsamples, nchains), algorithm,
    max_nsteps = 10 * nsamples,
    max_time = Inf,
    tuning = tuning,
    init = init,
    burnin = burnin,
    convergence = convergence,
    strict = false,
    filter = true
)
#md nothing # hide
#nb nothing # hide

# However, in many use cases, simply using the default options via
#
# ```julia
# samples, stats = bat_sample(posterior, (nsamples, nchains), MetropolisHastings())
# ```
#
# will often be sufficient.
