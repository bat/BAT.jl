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

# As the underlying truth of our input data/histogram, let us choose a
# non-normalized probability density composed of two Gaussian peaks with a peak
# area of 500 and 1000, a mean of -1.0 and 2.0, and a common standard error of 0.5.
# So our model parameters are:

par_names = ["a_1", "a_2", "mu_1", "mu_2", "sigma"]

true_par_values = [500, 1000, -1.0, 2.0, 0.5]
#md nothing # hide

# We'll define a Julia function that returns two Gaussian distributions, based
# on a specific set of parameters

function model_distributions(parameters::AbstractVector{<:Real})
    return (
        Normal(parameters[3], parameters[5]),
        Normal(parameters[4], parameters[5])
    )
end
#md nothing # hide

# [Fred] I was tempted to turn the julia parameters into unicode characters, too. But I guess newcomers might have a problem to enter the characters themself. But it would look so much nicer for a tutorial.

# and then generate some synthetic data by drawing a number (according to the
# parameters a₁ and a₂) of samples from the two Gaussian distributions

data = vcat(
    rand(model_distributions(true_par_values)[1], Int(true_par_values[1])),
    rand(model_distributions(true_par_values)[2], Int(true_par_values[2]))
)
#md nothing # hide

# [Fred] I don't see why this emphasis on floats is useful. If I run the previous line in the julia shell, I see the samples, so it's clear they are floats.

# resulting in a vector of floating-point numbers:

typeof(data) == Vector{Float64}

# Then we create a histogram of that data, this histogram will serve as the
# input for the Bayesian fit:

hist = append!(Histogram(-2:0.1:4), data)

# The fit function that describes such a histogram (depending on the model
# parameters) will be

function fit_function(x::Real, parameters::AbstractVector{<:Real})
    dists = model_distributions(parameters)
    return parameters[1] * pdf(dists[1], x) +
           parameters[2] * pdf(dists[2], x)
end
#md nothing # hide

# Using the Julia ["Plots"](http://docs.juliaplots.org/latest/) package

using Plots

# we can visually compare the histogram and the fit function, using the true
# values of the parameters:

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data and True Statistical Model"
)
plot!(
    -4:0.01:4, x -> fit_function(x, true_par_values),
    label = "Truth"
)
#jl savefig("tutorial-data-and-truth.pdf")
#md savefig("tutorial-data-and-truth.pdf")
#md savefig("tutorial-data-and-truth.svg"); nothing # hide
#md # [![Data and True Statistical Model](tutorial-data-and-truth.svg)](tutorial-data-and-truth.pdf)


# ## Bayesian Fit
#
# Now let's do a Bayesian fit of the generated histogram, using BAT.
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

# [FRED] It would be good to just render some math here, define the Poisson etc. You seem to assume everyone knows that but for non-particle physicists, don't take it for granted.

struct HistogramLikelihood{H<:Histogram} <: AbstractDensity
    histogram::H
end

# As a minimum, BAT requires methods of `BAT.nparams` and
# `BAT.unsafe_density_logval` to be defined for each subtype of
# `AbstractDensity`.
#
# `BAT.nparams` simply needs to return the number of free parameters:

BAT.nparams(likelihood::HistogramLikelihood) = 5

# `BAT.unsafe_density_logval` has to implement the actual log-likelihood
# function:
# [Fred] Some math symbol would be nice to explain the log likelihood. We had this discussion a long time ago: the name `unsafe_density_logval` can be a stopper to new users. Why does this package ask me to implement some "unsafe" function? Something like `core_density_logval` would be easier to swallow.

function BAT.unsafe_density_logval(
    likelihood::HistogramLikelihood,
    parameters::AbstractVector{<:Real},
    exec_context::ExecContext
)
    # [FRED] These comments inside the function are also rendered in the HTML output as ordinary text and make it hard to follow in HTML what the function is. Is there a way to say, "render this comment as an ordinary julia comment?". Else just leave these comments out, so a user can see the function as a whole.


    # [Fred] The code helped me more than the comment actually. I'd just drop the comment
    # Histogram counts for each bin as an array:
    counts = likelihood.histogram.weights

    # Histogram binning, has length (length(counts) + 1):
    binning = likelihood.histogram.edges[1]

    # [Fred] useless comment
    # sum log-likelihood over bins:
    log_likelihood::Float64 = 0.0
    for i in eachindex(counts)
        bin_left, bin_right = binning[i], binning[i+1]
        bin_width = bin_right - bin_left
        bin_center = (bin_right + bin_left) / 2

        observed_counts = counts[i]

        # Simple mid-point rule integration of fit_function over bin:
        expected_counts = bin_width * fit_function(bin_center, parameters)

        log_likelihood += logpdf(Poisson(expected_counts), observed_counts)
    end

    return log_likelihood
end


# Methods of `BAT.unsafe_density_logval` may be "unsafe" insofar as the
# implementation is not required to check the length of the `parameters` vector
# or the validity of the parameter values - BAT takes care of that (assuming
# that value provided by `BAT.nparams` is correct and that the prior that will
# only cover valid parameter values).
#
# Note: Currently, implementations of `BAT.unsafe_density_logval` *must* be
# type stable, to avoid triggering a Julia-internal error. The matter is under
# investigation. If the implementation of `BAT.unsafe_density_logval` is *not*
# type-stable, this will often result in an error like this:
#
# ```
# Internal error: encountered unexpected error in runtime:
# MethodError(f=typeof(Core.Compiler.fieldindex)(), args=(Random123.Philox4x{T, R} ...
# ```
# [fred] I'd rather link to the docs to explain what type stable means, and create an issue in `BAT.jl` public repo that you can link to as well. This tutorial is already long.
#
# The `exec_context` argument can be ignored in simple use cases, it is only
# of interest for `unsafe_density_logval` methods that internally use Julia's
# multi-threading and/or distributed code execution capabilities.
#
# BAT itself also makes use of Julia's parallel programming facilities. BAT
# can calculate log-density values in parallel (e.g. for multiple MCMC chains)
# on multiple threads (implemented) and support for distributed execution
# (on multiple hosts) is planned. By default, however, BAT will assume that
# implementations of `BAT.unsafe_density_logval` are *not* thread safe. If
# your implementation *is* thread-safe (as is the case in the example above),
# you can advertise this fact to BAT:

BAT.exec_capabilities(::typeof(BAT.unsafe_density_logval), likelihood::HistogramLikelihood, parameters::AbstractVector{<:Real}) =
    ExecCapabilities(0, true, 0, true)
# [Fred] the args `(0, true, 0, true)` are a bit hard to understand, it seems there is more than just thread-safety here

# BAT will then use multi-threaded log-likelihood evaluation where possible.
# Note that Julia starts only a single thread by default, you will need to set
# the environment variable
# [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1)
# to configure the number of Julia threads.
#
#
# Given our fit function and the histogram to fit, we'll define the
# likelihood as

likelihood = HistogramLikelihood(hist)


# ### Prior Definition
#
# For simplicity, we choose a flat prior, i.e. a normalized constant
# density, on the support made up by a hyper-rectangle:

prior = ConstDensity(
    HyperRectBounds(
        [
            0.0..10.0^4, 0.0..10.0^4,
            -2.0..0.0, 1.0..3.0,
            0.3..0.7
        ],
        reflective_bounds
    ),
    normalize
)
#md nothing # hide

# In general, BAT allows instances of any subtype of `AbstractDensity` to
# be used as a prior, as long as a sampler is defined for it. This way, users
# may implement complex application-specific priors. You can also
# use `convert(AbstractDensity, distribution)` to convert any
# continuous multivariate `Distributions.Distribution` to a
# `BAT.AbstractDensity` that can be used as a prior (or likelihood).


### Bayesian Model Definition
#
# Given the likelihood and prior definition, a `BAT.BayesianModel` is simply
# defined via

model = BayesianModel(likelihood, prior)
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

# We also need to specify which random number generator and seed to use. BAT requires
# a counter-based RNG and partitions the RNG space over the MCMC chains. This
# way, a single RNG seed is sufficient for all chains and results can be
# reproducible even under parallel execution. Let's choose a Philox4x RNG
# with a random seed:

rngseed = BAT.Philox4xSeed()
#md nothing # hide

# The algorithm, model, and RNG seed specify the MCMC chains:

chainspec = MCMCSpec(algorithm, model, rngseed)
#md nothing # hide

# Let's use 4  chains and require $10^5$ unique samples from each chain
# (after tuning/burn-in):

nsamples = 10^5
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

convergence_test = BGConvergence(1.1)

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

# Before running the Markov chains, let's set BAT's logging level to debug,
# to see what's going on in more detail (note: BAT's logging API will change
# in the future for better integration with the Julia v1 logging facilities):

BAT.Logging.set_log_level!(BAT, BAT.Logging.LOG_DEBUG)
#md nothing # hide

# Now we can generate a set of MCMC samples via `rand`:

samples, sampleids, stats, chains = rand(
    chainspec,
    nsamples,
    nchains,
    tuner_config = tuner_config,
    convergence_test = convergence_test,
    init_strategy = init_strategy,
    burnin_strategy = burnin_strategy,
    max_nsteps = 10000,
    max_time = Inf,
    granularity = 1,
    ll = BAT.Logging.LOG_INFO
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

# `stats` contains some statistics collected during MCMC sample generation, e.g.
# the mean and covariance of the parameters and the mode. These and any other
# statistics may of course be calculated from the samples after the sampling
# finished:

@assert vec(mean(samples.params, FrequencyWeights(samples.weight))) ≈ stats.param_stats.mean
# output warning
# ┌ Warning: `mean(A::AbstractArray{T}, w::AbstractWeights{W}, dims::Int) where {T <: Number, W <: Real}` is deprecated, use `mean(A, w, dims=dims)` instead.
# │   caller = mean(::ArraysOfArrays.ArrayOfSimilarArrays{Float64,1,1,2,ElasticArrays.ElasticArray{Float64,2,1}}, ::StatsBase.FrequencyWeights{Int64,Int64,Array{Int64,1}}) at statsbase_support.jl:6
# └ @ ArraysOfArrays ~/.julia/packages/ArraysOfArrays/Fd8FA/src/statsbase_support.jl:6
@assert vec(var(samples.params, FrequencyWeights(samples.weight))) ≈ diag(stats.param_stats.cov)
@assert cov(samples.params, FrequencyWeights(samples.weight)) ≈ stats.param_stats.cov

# We can also, e.g., get the Pearson auto-correlation of the parameters:

vec(cor(samples.params, FrequencyWeights(samples.weight)))


# ### Visualization of Results

# BAT.jl comes with an extensive set of plotting recipes for
# ["Plots.jl"] (http://docs.juliaplots.org/latest/).
# We can plot the marginalized distribution for a single parameter (e.g.
# parameter 3, i.e. μ₁):

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

#[Fred] Plots are always up to taste, here is my opinion.
# 1. You should have different symbols for the local and global mode, I see a black square for both.
# 2. The purple for the ellipses isn't pretty, and having the same color for all levels isn't good, either
# 3. In the legend, I would say 1\sigma instead of 68.46%, same for 2\sigma, 3\sigma. And the output digits should match, I see 68.30000000000001% instead of 68.3%

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

# [Fred] I see a lot of these messages in the rendered output but the matrix still shows up
# sh: 1: latex: not found
# latex: failed to create a dvi file

# Again my subjective judgment: the black marker on the half-black plots in the upper right triangle looks bad. Then the numbers on the ticks overlap, I think other packages solve this by rotating the labels by 45 degrees.

# ### Integration with Tables.jl

# BAT.jl supports the [Tables.jl](https://github.com/JuliaData/Tables.jl)
# interface. Using a tables implementation like
# [TypedTables.jl](http://blog.roames.com/TypedTables.jl/stable/),
# the whole MCMC output (parameter vectors, weights, sample/chain numbers,
# etc.) can easily can be combined into a single table:

# [Fred] This is really nice, people don't have to learn anything new to analyze BAT samples. But it this just clever iterators or actually copying data?

using TypedTables

tbl = Table(samples, sampleids)

# We can now, e.g., find the sample with the maximum posterior value (i.e. the
# mode):

mode_log_posterior, mode_idx = findmax(tbl.log_posterior)

# And get row `mode_idx` of the table, with all information about the sample
# at the mode:

# [Fred] Looks like some code examples is missing for mode_idx

# ## Comparison of Truth and Best Fit

# As a final step, we retrieve the parameter values at the mode, representing
# the best-fit parameters

fit_par_values = tbl[mode_idx].params

# And plot the truth, data, and best fit:

plot(
    normalize(hist, mode=:density),
    st = :steps, label = "Data",
    title = "Data, True Model and Best Fit"
)
plot!(-4:0.01:4, x -> fit_function(x, true_par_values), label = "Truth")
plot!(-4:0.01:4, x -> fit_function(x, fit_par_values), label = "Best fit")
#jl savefig("tutorial-data-truth-bestfit.pdf")
#md savefig("tutorial-data-truth-bestfit.pdf")
#md savefig("tutorial-data-truth-bestfit.svg"); nothing # hide
#md # [![Data, True Model and Best Fit](tutorial-data-truth-bestfit.svg)](tutorial-data-truth-bestfit.pdf)

# [Fred] The best-fit comparison is nice but where is the uncertainty band? Else why run all the chains, you could have had this plot alone with max-likelihood.
