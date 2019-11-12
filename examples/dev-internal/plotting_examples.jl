#md # # Plot recipes examples
#nb # # BAT.jl Plot recipes examples
#jl # # BAT.jl Plot recipes examples


using BAT

# ## Generate samples
# We generate some generic multimodal samples to be plotted afterwards
using Distributions, IntervalSets

##Model definition to generate samples
struct GaussianShellDensity<:AbstractDensity
    r::Vector{Float64}
    sigma::Vector{Float64}
end

#define likelihood
function BAT.density_logval(target::GaussianShellDensity, params::Any)
    loglikelihood::Float64 = 0.
    for i in 1:length(params)
        
        result = -0.5 * (params[i][1]-target.r[i])^2/target.sigma[i]^2
        l1 = result - log(√2π * target.sigma[i])
        
        result2 = -0.5 * (params[i][1]+target.r[i])^2/target.sigma[i]^2
        l2 = result2 - log(√2π * target.sigma[i])^2
        
        loglikelihood += log(exp(l1) + 2*exp(l2)) 
    end
    
    return loglikelihood
end

likelihood = GaussianShellDensity([5.0, 5., 3.], [2, 2.4, 1.5]) 

prior = NamedTupleDist(
    λ1 = Normal(-3, 4.5),
    λ2 = -30.0..30.0,    
    λ3 = Normal(3, 3.5)    
)

posterior = PosteriorDensity(likelihood, prior);

#generate samples
samples, stats, chains = bat_sample(posterior, (10^5, 4), MetropolisHastings())


# ## Set up plotting
#  We set up plotting by using the ["Plots"](https://github.com/JuliaPlots/Plots.jl) package with the pyplot backend.
using Plots

# The plotting recipes have been tested with `pyplot`, `gr` and `plotly` backends.
# All available plotting options work correctly with `pyplot`.
# Only the colored 2D contour plots are currently not correctly supported with `gr` or `plotly` backends.

pyplot()

# ## Examples for 1D plots 
# Below, all available seriestypes and plotting features for 1D representations of samples or prior are shown
# ### default 1D plot style for samples:
plot(samples, 2)
# The default seriestype for samples is `:smallest_intervals`.
# By default, the local mode of the histogram is indicated as a dotted black line.

# ### default 1D plot style for prior:
plot(prior, 2)
# The default seriestype for sprior is `:stephist`.

# Samples can either be plotted by their index (as shown above) or by using the parameter names given in NamedTupleDist. This can be done by passing either the posterior 
plot(posterior, samples, :λ2)
# or the value shape(s) of the prior:
using ValueShapes
parshapes = valshape(prior)
plot(parshapes, samples, :λ2)

# Prior can also be plotted by their index or by using the parameter names given in NamedTupleDist:
plot(prior, 1)
# or
plot(prior, :λ1)

# Plot prior + posterior
plot(posterior, samples, :λ1)
plot!(prior, :λ1)

# ### Further seriestypes available:
# #### central intervals
plot(samples, 2, seriestype = :central_intervals)

# ### Customizing smallest/central interval plots:
# The probability intervals to be highlighted can be specified using the `intervals` keyword. Their colors (in same order) need to be specified using the `colors` keyword argument.
plot(samples, 2, seriestype=:smallest_intervals, intervals=[0.5, 0.1, 0.3, 0.99], colors=[:grey, :red, :blue, :orange])

# #### histogram
plot(samples, 2, seriestype = :histogram)  
# alias: `:hist == :histogram`

# #### step histogram
plot(samples, 2, seriestype = :stephist)

# ### Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can be passed:
plot(samples, 2, seriestype = :stephist, nbins=50, linecolor = :red, linewidth = 5, linealpha=0.3, xlim=(0,10))

# ### Plotting point estimators in 1D plots:
plot(samples, 2, globalmode=true, localmode=true, mean=true, std_dev=true)

# #### It is possible to customize the style of estimators by passing a dict with the respective attributes:
# By passing a boolean, the point estimators are plotted using their default styles shown above.
# The style of the point estimators can be modified by passing a dictionary specifying `linestyle`, `linecolor`, `linewidth` and `alpha` for *mean*, *globalmode* and *localmode*.
# The style of the standard deviation can be modified by specifying `fillcolor` and `fillalpha`.
plot(samples, 1, localmode=false, mean=Dict("linestyle" => :dot, "linecolor"=> :red, "linewidth"=>2, "alpha" => 0.7), std_dev=Dict("fillcolor" => :red, "fillalpha"=>0.2))

# ## Examples for 2D plots of the samples
# Below, all available seriestypes and plotting features for 2D representations of samples are shown
# ### default 2D plot style:
plot(samples, (1,2), nbins=200)
# The default 2D plotting style is a 3-color heatmap showing the smallest intervals containing 68.3, 95.5 and 99.7 perecent of the posterior probability. 

# Samples can either be plotted by their index (as shown above) or by using the parameter names given in NamedTupleDist. This can be done by passing either the posterior 
plot(posterior, samples, (:λ1, :λ2))
# or the value shape(s) of the prior:
parshapes = valshape(prior)
plot(parshapes, samples, (:λ1, :λ2))

# Prior can also be plotted by their index or by using the parameter names given in NamedTupleDist.
plot(prior, (1, 2))
# or
plot(prior, (:λ1, :λ2))


# Plot prior + posterior
plot(posterior, samples, (:λ1, :λ3))
plot!(prior, (:λ1, :λ3))

# ### Further seriestypes available:
# #### central intervals
plot(samples, (1,2), seriestype = :histogram, nbins=200)
# alias: `:histogram2d == :histogram`

# #### smallest intervals as colored contour lines: 
# (currently only correctly supported with `pyplot()` backend)
plot(samples, (1,2), seriestype=:smallest_intervals_contour, bins=50)

# #### smallest intervals as filled contours: 
# (currently only correctly supported with `pyplot()` backend)
plot(samples, (1,2), seriestype=:smallest_intervals_contourf, bins=50)

# ### Customizing smallest interval plots:
# The probability intervals to be highlighted can be specified using the `intervals` keyword. Their colors (in same order) need to be specified using the `colors` keyword argument.
plot(samples, (1,2), seriestype=:smallest_intervals, nbins=200, intervals=[0.7, 0.2], colors=[:blue, :red])

# #### scatter
# for large number of samples, this plotting style needs a lot of time and resources
#plot(samples, (1,2), seriestype = :scatter)

# ### Plotting point estimators in 2D plots:
plot(samples, (1,2), seriestype=:smallest_intervals, nbins=200, mean=true, std_dev=true, localmode=true, globalmode=true)

# #### It is possible to customize the style of estimators by passing a dict with the respective attributes:
# By passing a boolean, the point estimators are plotted using their default styles shown above.
# The style of the point estimators can be modified by passing a dictionary specifying `markershape`, `markercolor`, `markersize`, `markeralpha`, `markerstrokecolor`, `markerstrokestyle`, `markerstrokewidth` and `markerstrokealpha` for *mean*, *globalmode* and *localmode*.
# If `std_dev==true`, the standard deviation of the mean value will be displayed as x- and y-errorbars.
plot(samples, (1,2), seriestype=:smallest_intervals, nbins=200, localmode=Dict("markershape"=> :diamond, "markeralpha"=>1, "markercolor"=>:red, "markersize"=>5))


# ### Marginal plots
plot(samples, (1,2), seriestype = :marginal)

# ### Customizing marginal plots:
# The marginal plots can be modified by passing dictionaries to the keyword arguments `upper`, `right` and `diagonal`.
# The dictionaries for `upper` and `right` can contain the 1D seriestypes and plot options shown above.
# The dictionary for `diagonal` can use the 2D seriestypes and plot options shown above.
# Nested dictonaries are possible (e.g. for modifying point estimators)
plot(samples, (1,2), seriestype = :marginal, diagonal = Dict("seriestype"=>:histogram, "mean"=>Dict("markercolor"=>:green)), upper=Dict("mean" => true, "seriestype" => :smallest_intervals, "colors"=>[:blue, :grey, :orange]), right=Dict("seriestype" => :stephist, "mean"=>true))


# ### Overview plot
plot(samples)
#
plot(prior)
#
plot(samples)
plot!(prior)

# By default the 1D and 2D plots for the first 5 parameters are shown.
# The keyword argument `params` allows to specify the considered parameters.
plot(samples, params=[1, 3])


# ### Customizing overview plots:
# The overview plots can be modified by passing dictionaries to the keyword arguments `upper`, `lower` and `diagonal`.
# The dictionaries for `upper` and `lower` can contain the 2D seriestypes and plot options shown above.
# The dictionary for `diagonal` can use the 1D seriestypes and plot options shown above.
# Nested dictonaries are possible (e.g. for modifying point estimators)

plot(samples, mean=true, globalmode=true, legend=true, diagonal=Dict("seriestype"=>:stephist, "mean"=>Dict("linecolor" => :green, "linewidth" => 8)), lower = Dict("mean" => false, "colors"=>[:orange, :green, :grey]))


# ## Plots for MCMC diagnostics
# Plots histograms of the samples, the trace, a kernel density estimate and the autocorrelation function for each parameter per chain.
diagnostics = BAT.MCMCDiagnostics(samples, chains)
plot(diagnostics, params=[1])

# ### Customizing diagnostics plots:
plot(diagnostics, 
    params=[1, 2], 
    chains=[1, 2], 
    diagnostics = [:histogram, :kde, :trace, :acf],
    histogram = Dict("seriestype" => :smallest_intervals, "legend" => :false),
    trace = Dict("linecolor" => :red),
    acf = Dict("lags" => collect(1:20), "title"=> "Autocorrelation"),
    description = true)

# ### available keyword arguments:
# * `params` - list of parameters to be plotted
# * `chains` - list of chains to be plotted
# * `diagnostics` - list of MCMC diagnostics to be plotted
# * `:histogram` - 1D histograms of samples
# * `:kde` - Kernel density estimate (using [*KernelDensity.jl*](https://github.com/JuliaStats/KernelDensity.jl))
# * `:trace` - Trace plot
# * `:acf` - Autocorrelation function (using [*StatsBase.autocor*](http://juliastats.github.io/StatsBase.jl/stable/signalcorr/#StatsBase.autocor))
# * `description::Bool = true` - show description (current chain, parameter, number of samples) as first column of plots
# * `histogram::Dict` - options for histogram plots (supports all arguments for 1D plots for samples)
# * `kde::Dict` - options for kde plots
# * `trace::Dict` - options for trace plots
# * `acf::Dict` - options for acf plots

# ### special options arguments for `:kde` (see [*KernelDensity.jl*](https://github.com/JuliaStats/KernelDensity.jl))
# * `npoints`: number of interpolation points to use (default: npoints = 2048)
# * `boundary`: lower and upper limits of the kde as a tuple
# * `kernel`: the distributional family from [*Distributions.jl*](https://github.com/JuliaStats/Distributions.jl) to use as the kernel (default = Distributions.Normal)
# * `bandwidth`: bandwidth of the kernel

# ### special keyword arguments for `:acf` (see [*StatsBase.autocor*](http://juliastats.github.io/StatsBase.jl/stable/signalcorr/#StatsBase.autocor))
# * `lags` - list of lags to be considered for ACF plots
# * `demean` - denotes whether the mean should be subtracted before computing the ACF