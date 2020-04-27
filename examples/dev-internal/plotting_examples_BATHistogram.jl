#md # # Plot recipes examples
#nb # # BAT.jl Plot recipes examples
#jl # # BAT.jl Plot recipes examples

using BAT
using StatsBase
using Distributions

# ## Generate a BATHistogram
data1 = rand(MixtureModel(Normal[
   Normal(-10.0, 1.2),
   Normal(0.0, 1.0),
   Normal(10.0, 2.5)], [0.2, 0.5, 0.3]), 100_000)

data2 = rand(MixtureModel(Normal[
   Normal(-5.0, 1.2),
   Normal(5.0, 2.5)], [0.3, 0.7]), 100_000)

bathist = BATHistogram(fit(Histogram, (data1, data2), closed=:left, nbins=20))

mean(bathist.h.weights)


# ## Set up plotting
#  Set up plotting by using the ["Plots"](https://github.com/JuliaPlots/Plots.jl) package with the pyplot backend.
using Plots

# The plotting recipes of BAT.jl have been tested with `pyplot`, `gr` and `plotly` backends.
# All available plotting options work correctly with `pyplot`.
# Currently, only the colored 2D contour plots are not correctly supported with `gr` or `plotly` backends.

gr()

# ## Examples for 1D plots
# Below, all available seriestypes and plotting options for 1D representations of a BATHistogram are shown

# #### step histogram (default)
plot(bathist, 1, seriestype = :stephist)

# ### smallest intervals
# The highest density region (HDR) is highlighted
plot(bathist, 1, seriestype = :smallest_intervals)
# alias: `:smallest_intervals == :HDR`

# ### central intervals
plot(bathist, 1, seriestype = :central_intervals)

# ### Customizing smallest/central interval plots:
# The relative probability to be enclosed in the intervals can be specified using the `intervals` keyword.
# The corresponding colors for the intervals need to be specified using the `colors` keyword argument in same order.
plot(bathist, 1, seriestype=:smallest_intervals, intervals=[0.5, 0.1, 0.3, 0.99], colors=[:grey, :red, :blue, :orange])

# #### filled histogram
plot(bathist, 2, seriestype = :histogram)
# alias: `:hist == :histogram`

# ### Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can be passed, e.g.:
plot(bathist, 2, seriestype = :stephist, nbins=50, linecolor = :red, linewidth = 5, linealpha=0.3, xlim=(-5,15))

# TODO: claculate point estimates for BATHistograms?


# ## Examples for 2D plots of BATHistograms
# Below, all available seriestypes and plotting features for 2D representations of BATHistograms are shown
# ### default 2D histogram style:
plot(bathist, (1,2))
# alias: `:histogram2d == :histogram`

# ### smallest intervals / HDR:
plot(bathist, (1,2), seriestype=:smallest_intervals)

# ### smallest intervals as colored contour lines:
# (currently only correctly supported with `pyplot()` backend)
plot(bathist, (1,2), seriestype=:smallest_intervals_contour)

# ### smallest intervals as filled contours:
# (currently only correctly supported with `pyplot()` backend)
plot(bathist, (1,2), seriestype=:smallest_intervals_contourf)

# ### Customizing smallest interval plots:
# The relative probability to be enclosed in the intervals can be specified using the `intervals` keyword.
# The corresponding colors for the intervals need to be specified using the `colors` keyword argument in same order.
plot(bathist, (1,2), seriestype=:smallest_intervals, intervals=[0.7, 0.2], colors=[:blue, :red])


# TODO: Point estimates for 2D BATHistograms


# ### Marginal plots
#TODO: why is there an error? it somehow still works for samples...
#plot(bathist, (1,2), seriestype = :marginal)


# ### Customizing marginal plots:
# The marginal plots can be modified by passing dictionaries to the keyword arguments `upper`, `right` and `diagonal`.
# The dictionaries for `upper` and `right` can contain the 1D seriestypes and plot options shown above.
# The dictionary for `diagonal` can use the 2D seriestypes and plot options shown above.
# Nested dictonaries are possible (e.g. for modifying point estimators)
plot(samples, (1,2), seriestype = :marginal, diagonal = Dict("seriestype"=>:histogram, "mean"=>Dict("markercolor"=>:green)), upper=Dict("mean" => true, "seriestype" => :smallest_intervals, "colors"=>[:blue, :grey, :orange]), right=Dict("seriestype" => :stephist, "mean"=>true))

# TODO: Overview for BATHistogram
# # ### Overview plot
# plot(samples)
# #
# plot(prior)
# #
# plot(samples)
# plot!(prior)
#
# # By default the 1D and 2D plots for the first 5 parameters are shown.
# # The keyword argument `vsel` allows to specify the considered parameters.
# plot(samples, vsel=[1, 3])
#
#
# # ### Customizing overview plots:
# # The overview plots can be modified by passing dictionaries to the keyword arguments `upper`, `lower` and `diagonal`.
# # The dictionaries for `upper` and `lower` can contain the 2D seriestypes and plot options shown above.
# # The dictionary for `diagonal` can use the 1D seriestypes and plot options shown above.
# # Nested dictonaries are possible (e.g. for modifying point estimators)
#
# plot(samples, mean=true, globalmode=true, legend=true, diagonal=Dict("seriestype"=>:stephist, "mean"=>Dict("linecolor" => :green, "linewidth" => 8)), lower = Dict("mean" => false, "colors"=>[:orange, :green, :grey]))
#
