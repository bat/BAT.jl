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
# ["Distributions"](https://juliastats.org/Distributions.jl/stable/)
# and ["StatsBase"](https://juliastats.org/StatsBase.jl/stable/):

using Random, LinearAlgebra, Statistics, Distributions, StatsBase

# As the underlying truth of our input data/histogram, let us choose the
# expected count to follow the sum of two Gaussian peaks with peak
# areas of 500 and 1000, a mean of -1.0 and 2.0 and a standard error of 0.5.
# Then 

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

true_par_values = (a = [500, 1000], mu = [-1.0, 2.0], sigma = 0.5)
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

using BAT, DensityInterface, IntervalSets


# ### Likelihood Definition
#
# First, we need to define the likelihood for our problem.
#
# BAT expects likelihoods to implements the `DensityInterface` API. We
# can simply wrap a log-likelihood function with
# `DensityInterface.logfuncdensity` to make it compatible.
#
# For performance reasons, functions should [not access global variables
# directly] (https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Avoid-global-variables-1).
# So we'll use an [anonymous function](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions-1)
# inside of a [let-statement](https://docs.julialang.org/en/v1/base/base/#let)
# to capture the value of the global variable `hist` in a local variable `h`
# (and to shorten function name `fit_function` to `f`, purely for
# convenience). `DensityInterface.logfuncdensity` then turns the
# log-likelihood function into a `DensityInterface` density object.

likelihood = let h = hist, f = fit_function
    ## Histogram counts for each bin as an array:
    observed_counts = h.weights

    ## Histogram binning:
    bin_edges = h.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    logfuncdensity(function (params)
        ## Log-likelihood for a single bin:
        function bin_log_likelihood(i)
            ## Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_widths[i] * f(params, bin_centers[i])
            ## Avoid zero expected counts for numerical stability:
            logpdf(Poisson(expected_counts + eps(expected_counts)), observed_counts[i])
        end

        ## Sum log-likelihood over bins:
        idxs = eachindex(observed_counts)
        ll_value = bin_log_likelihood(idxs[1])
        for i in idxs[2:end]
            ll_value += bin_log_likelihood(i)
        end

        return ll_value
    end)
end

# BAT makes use of Julia's parallel programming facilities if possible, e.g.
# to run multiple Markov chains in parallel. Therefore, log-likelihood
# (and other) code must be thread-safe. Mark non-thread-safe code with
# `@critical` (provided by Julia package `ParallelProcessingTools`).
#
# Support for automatic parallelization across multiple (local and remote)
# Julia processes is planned, but not implemented yet.
#
# Note that Julia currently starts only a single thread by default. Set the
# the environment variable
# [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1)
# to specify the desired number of Julia threads.

# We can evaluate `likelihood`, e.g. at the true parameter values:

logdensityof(likelihood, true_par_values)


# ### Prior Definition
#
# Next, we need to choose a sensible prior for the fit:

prior = distprod(
    a = [Weibull(1.1, 5000), Weibull(1.1, 5000)],
    mu = [-2.0..0.0, 1.0..3.0],
    sigma = Weibull(1.2, 2)
)

#md nothing # hide

# BAT supports most `Distributions.Distribution` types, and combinations
# of them, as priors.


# ### Bayesian Model Definition
#
# Given the likelihood and prior definition, a `BAT.PosteriorMeasure` is simply
# defined via

posterior = PosteriorMeasure(likelihood, prior)
#md nothing # hide


# ### Parameter Space Exploration via MCMC
#
# We can now use Markov chain Monte Carlo (MCMC) to explore the space of
# possible parameter values for the histogram fit.
#
# To increase the verbosity level of BAT logging output, you may want to set
# the Julia logging level for BAT to debug via `bat_logdebug()`.

#nb bat_logdebug()
#jl bat_logdebug()

# Now we can generate a set of MCMC samples via [`bat_sample`](@ref). We'll
# use 4 MCMC chains with 10^5 MC steps in each chain (after tuning/burn-in):

samples = bat_sample(posterior, TransformedMCMC(proposal = RandomWalk(), nsteps = 10^5, nchains = 4)).result
#md nothing # hide
#nb nothing # hide


# Let's calculate some statistics on the posterior samples:

println("Truth: $true_par_values")
println("Mode: $(mode(samples))")
println("Mean: $(mean(samples))")
println("Stddev: $(std(samples))")

# Internally, BAT often needs to represent variates as flat real-valued
# vectors:

unshaped_samples, f_flatten = bat_transform(Vector, samples)

# The statisics above (mode, mean and std-dev) are presented in shaped form.
# However, it's not possible to represent statistics with matrix shape, e.g.
# the parameter covariance matrix, this way. So the covariance has to be
# accessed in unshaped form:

par_cov = cov(unshaped_samples)
println("Covariance: $par_cov")


# Use `LazyReports.lazyreport` to generate an overview of the sampling result and parameter estimates (based on the marginal distributions):

using LazyReports
lazyreport(samples)


# ### Visualization of Results

# BAT.jl comes with an extensive set of plotting recipes for
# ["Plots.jl"] (http://docs.juliaplots.org/latest/).
# We can plot the marginalized distribution for a single parameter (e.g.
# parameter 3, i.e. μ[1]):

plot(
    samples, :(mu[1]),
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50, title = "Marginalized Distribution for mu[1]"
)
#jl savefig("tutorial-single-par.pdf")
#md savefig("tutorial-single-par.pdf")
#md savefig("tutorial-single-par.svg"); nothing # hide
#md # [![Marginalized Distribution for mu_1](tutorial-single-par.svg)](tutorial-single-par.pdf)

# or plot the marginalized distribution for a pair of parameters (e.g.
# parameters 3 and 5, i.e. μ[1] and σ), including information from the parameter
# stats:

plot(
    samples, (:(mu[1]), :sigma),
    mean = true, std = true, globalmode = true, marginalmode = true,
    nbins = 50, title = "Marginalized Distribution for mu[1] and sigma"
)
plot!(BAT.MCMCBasicStats(samples), (3, 5))
#jl savefig("tutorial-param-pair.png")
#md savefig("tutorial-param-pair.png")
#md savefig("tutorial-param-pair.svg"); nothing # hide
#md # [![Marginalized Distribution for mu_1 and sigma](tutorial-param-pair.svg)](tutorial-param-pair.png)

# We can also create an overview plot of the marginalized distribution for all
# pairs of parameters:

plot(
    samples,
    mean = false, std = false, globalmode = true, marginalmode = false,
    nbins = 50
)
#jl savefig("tutorial-all-params.png")
#md savefig("tutorial-all-params.png")
#md savefig("tutorial-all-params.svg"); nothing # hide
#md # [![Pairwise Correlation between Parameters](tutorial-all-params.svg)](tutorial-all-params.png)


# ### Integration with Tables.jl

# `DensitySamplesVector` supports the
# [Tables.jl](https://github.com/JuliaData/Tables.jl)
# interface, so it is a table itself. We can also convert it to other table
# types, e.g. a
# [`TypedTables.Table`](http://blog.roames.com/TypedTables.jl/stable/):

using TypedTables

tbl = Table(samples)

# or a [`DataFrames.DataFrame`](https://github.com/JuliaData/DataFrames.jl),
# etc.


# ## Comparison of Truth and Best Fit

# As a final step, we retrieve the parameter values at the mode, representing
# the best-fit parameters

samples_mode = mode(samples)

# Like the samples themselves, the result can be viewed in both shaped and
# unshaped form. `samples_mode` is presented as a 0-dimensional array that
# contains a NamedTuple, this representation preserves the shape information:

samples_mode isa NamedTuple

# `samples_mode` is only an estimate of the mode of the posterior
# distribution. It can be further refined using [`bat_findmode`](@ref):

using Optim

findmode_result = bat_findmode(
    posterior,
    OptimAlg(optalg = Optim.NelderMead(), init = ExplicitInit([samples_mode]))
)

fit_par_values = findmode_result.result


# Let's plot the data and fit function given the true parameters and MCMC samples

plot(-4:0.01:4, fit_function, samples)

plot!(
    normalize(hist, mode=:density),
    color=1, linewidth=2, fillalpha=0.0,
    st = :steps, fill=false, label = "Data",
    title = "Data, True Model and Best Fit"
)

plot!(-4:0.01:4, x -> fit_function(true_par_values, x), color=4, label = "Truth")
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

# We'll sample using the random-walk Metropolis-Hastings MCMC algorithm:

mcmcalgo = RandomWalk()

# BAT requires a counter-based random number generator (RNG), since it
# partitions the RNG space over the MCMC chains. This way, a single RNG seed
# is sufficient for all chains and results are reproducible even under
# parallel execution. By default, BAT uses a Philox4x RNG initialized with a
# random seed drawn from the
# [system entropy pool](https://docs.julialang.org/en/v1/stdlib/Random/index.html#Random.RandomDevice):

using Random123
rng = Philox4x()
context = BATContext(rng = Philox4x())
#md nothing # hide


# By default, `RandomWalk()` uses the following options.
#
# For Markov chain initialization:

init = MCMCChainPoolInit()

# For the MCMC burn-in procedure:

burnin = MCMCMultiCycleBurnin()

# For convergence testing:

convergence = BrooksGelmanConvergence()

# To generate MCMC samples with explicit control over all options, use
# something like

samples = bat_sample(
    posterior,
    TransformedMCMC(
        proposal = mcmcalgo,
        nchains = 4,
        nsteps = 10^5,
        init = init,
        burnin = burnin,
        convergence = convergence,
        strict = true,
        store_burnin = false,
        nonzero_weights = true,
        callback = (x...) -> nothing
    ),
    context
).result
#md nothing # hide
#nb nothing # hide

# ## Saving result data to files
#
# The package [FileIO.jl](https://github.com/JuliaIO/FileIO.jl)(in conjunction
# with [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)) offers a convenient way
# to store results like posterior samples to file:
#
# ```julia
# using FileIO
# import JLD2
# FileIO.save("results.jld2", Dict("samples" => samples))
# ```
#
# JLD2 persists the full information (including value shapes), so you can
# reload exactly the same data into memory in a new Julia session via
#
# ```julia
# using FileIO
# import JLD2
# samples = FileIO.load("results.jld2", "samples")
# ```
#
# provided you use compatible versions of BAT and it's dependencies. Note that
# JLD2 is *not* a long-term stable file format. Also note that this functionality
# is provided by FileIO.jl and JLD2.jl and not part of the BAT API itself.
#
# BAT.jl itself can write samples to standard HDF5 files in a form suitable for
# long-term storage (via [HDF5.jl](https://github.com/JuliaIO/HDF5.jl)):
#
# ```julia
# import HDF5
# bat_write("results.h5", samples)
# ```
#
# The resulting files have an intuitive HDF5 layout and can be read with the
# standard HDF5 libraries, so they are easily accessible from other programming
# languages as well. Not all value shape information can be preserved, though.
# To read BAT.jl HDF5 sample data, use
#
# ```julia
# using BAT
# import HDF5
# samples = bat_read("results.h5").result
# ```
#
# BAT.jl's HDF5 file format may evolve over time, but future versions of BAT.jl
# will be able to read HDF5 sample data written by this version of BAT.jl.
