# Plot Recipes
The plot recipes described below apply for plotting samples and priors. Only the plotting of estimators (mean, standard deviation, global mode and local mode) is currently only supported for samples.

Also see the plotting tutorial for examples and further information on the plotting options.

## 1D plots
``` julia
plot(
    samples::DensitySampleVector / prior::NamedTupleDist,
    vsel::Union{Integer, Symbol, Expr};
    intervals = BAT.default_credibilities,
    bins = 200,
    colors = BAT.default_colors,
    interval_labels = [],
    mean = false,
    std = false,
    globalmode = false,
    marginalmode = true,
    filter = false,
    closed = :left
)
```
Required inputs:
  * `samples::DensitySampleVector` or `prior::NamedTupleDist`: samples (shaped or unshaped) or prior to be plotted
  * `vsel::Union{Integer, Symbol, Expr}`: index or name of the parameter to be plotted

Keyword arguments:
  * `seriestype::Symbol = :smallest_intervals`: plot style        
	available seriestypes:
	* `:smallest_intervals` (default for samples), alias `:HDR`
    * `:central_intervals`
    * `:histogram`, alias `:steppost`
    * `:stephist` (default for prior)

* `bins::Union{Integer, AbstractRange} = 200`: number of histogram bins or bin edges.

* `intervals::Array{<:Real, 1} = [0.683, 0.955, 0.997]`: probability to be enclosed in the smallest/central intervals when using the corresponding seriestypes

* `colors::Array{Symbol, 1} = [:chartreuse2, :yellow, :red]`: colors of the smallest/central interval regions, in same order as the values in `intervals`

* `mean::Union{Dict, Bool} = true`: indicate mean value (currently only for samples), calculated via `bat_stats().mean`.
The plot style of the mean can be customized using a `Dict`.  For `mean = true`, the default style is:  
`Dict("linestyle" => :solid, "linewidth" => 1, "linecolor" => :dimgrey, "alpha" => 1, "label" => "mean")`

* `std::Union{Dict, Bool} = true`: indicate the standard deviation of the mean (currently only for samples), calculated from `bat_stats().cov`. The style of the standard deviation can be customized using a `Dict`.  For `std = true`, the default style is:   
`Dict("fillcolor" => :grey, "fillalpha" => 0.5, "label" => "std. dev.")`

* `globalmode::Union{Dict, Bool} = true`: indicate global mode (currently only for samples), calculated via `bat_stats().mode`.  The style of the global mode can be passed as a `Dict`.  For `globalmode = true`, the default style is:  
`Dict("linestyle" => :dash, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "global mode")`

* `marginalmode::Union{Dict, Bool} = true`: indicate the marginal mode(s), i.e. the center of the highest histogram bin(s) (currently only for samples). The style can be passed as a `Dict`. If `marginalmode = true`, the default style is:  
`Dict("linestyle" => :dot, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "local mode")`

* (only for samples) `filter::Bool = false`: if `true`, `BAT.drop_low_weight_samples()` is applied before plotting

* `closed = :left`: see [StatsBase.Histogram](https://juliastats.org/StatsBase.jl/stable/empirical/#StatsBase.fit-Tuple{Type{Histogram},Vararg{Any,N}%20where%20N})

Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can also be passed to the function.



## 2D plots
``` julia
plot(
    samples::DensitySampleVector / prior::NamedTupleDist,
    vsel::Union{NTuple{2, Integer}, NTuple{2, Union{Symbol, Expr}}};
    intervals = BAT.default_credibilities,
	interval_labels = [],
    colors = BAT.default_colors,
	bins = 200,
    mean = false,
    std = false,
    globalmode = false,
    marginalmode = true,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = false,
    closed = :left
)
```
Required inputs:
  * `samples::DensitySampleVector` or `prior::NamedTupleDist`: samples (shaped or unshaped) or prior to be plotted
  * `vsel::Union{NTuple{2, Integer}, NTuple{2, Union{Symbol, Expr}}}`: indices or names of the two parameters to be plotted

Keyword arguments:
  * `seriestype::Symbol = :smallest_intervals`: plot style  
	available seriestypes:
	* `:smallest_intervals` (default for samples)
	* `:smallest_intervals_contour` (default for prior)
	* `:smallest_intervals_contourf`: filled contours)
	* `:histogram`, alias `:hist`, alias `:histogram2d`
    * `:scatter`


* `bins::Union{Integer, NTuple{2, Integer}, NTuple{2, AbstractRange}} = 200`: number of histogram bins or bin edges. Use a `NTuple{2, Union{Integer, AbstractRange}}` to specify bins/edges of x and y axes seperately.

* `intervals::Array{<:Real, 1} = [0.683, 0.955, 0.997]`  probability to be enclosed in the smallest intervals when using the corresponding seriestypes

* `interval_labels = []`  label of the intervals as strings, `LatexStrings` are possible.

* `colors::Array{Symbol, 1} = [:chartreuse2, :yellow, :red]` colors of the smallest interval regions, in same order as the values in `intervals`

* `mean::Union{Dict, Bool} = true`: indicate mean value (currently only for samples) , calculated via `bat_stats().mean`.
The plot style of the mean can be customized using a `Dict`.  For `mean = true`, the default style is:
`Dict("markercolor" => :black, "markersize" => 4, "markeralpha" => 1, "markershape" => :circle, "markerstrokealpha" => 1, "markerstrokecolor" => :black, "markerstrokestyle" => :solid, "markerstrokewidth" => 1, "label" => "mean")`

* `std::Union{Bool} = true`: indicate the standard deviation of the mean as errorbars (currently only for samples), calculated from `bat_stats().cov`. The style of the errorbars can be customized using the `markerstroke...` options in `mean`.

* `globalmode::Union{Dict, Bool} = true`: indicate global mode (currently only for samples), calculated via `bat_stats().mode`.  The style of the global mode can be passed as a `Dict`.  For `globalmode = true`, the default style is:
`Dict("linestyle" => :dash, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "global mode")`

* `marginalmode::Union{Dict, Bool} = true`: indicate the marginalmode(s), i.e. the center of the highest histogram bin(s) (currently only for samples). The style can be passed as a `Dict`. If `marginalmode = true`, the default style is:
`Dict("linestyle" => :dot, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "local mode")`

*  `diagonal = Dict()`: Used only for the seriestype `:marginal`. The dictionary can contain the seriestypes and plot options for 2D distributions explained above to modify the 2D plot of the marginal plot. Nested Dictionaries are possible to modify the styles of the estimators as described above

* `upper = Dict()` and `right = Dict()`:  Used only for the seriestype `:marginal`. The dictionaries can contain the seriestypes and plot options for 1D distributions explained above to modify the upper and right plots of the 1D marginal distributions. Nested Dictionaries are possible to modify the styles of the estimators as described above

* `filter::Bool = false`: if `true`, `BAT.drop_low_weight_samples()` is applied before plotting

* `closed = :left`: see [StatsBase.Histogram](https://juliastats.org/StatsBase.jl/stable/empirical/#StatsBase.fit-Tuple{Type{Histogram},Vararg{Any,N}%20where%20N})

Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can also be passed to the function.


## 1D & 2D overview plots
``` julia
plot(
	samples::DensitySampleVector / prior::NamedTupleDist;
    vsel=collect(1:5),
	bins = 200,
    mean=false,
    std=false,
    globalmode=false,
    marginalmode=false,
    diagonal = Dict(),
    upper = Dict(),
    lower = Dict(),
    vsel_label = []
)
```
Plot a grid with the 1D marginalized distributions on the diagonal and all combinations of the 2D marginalized distributions in the lower and upper triangle.

Required inputs:
  * `samples::DensitySampleVector` or `prior::NamedTupleDist`: samples (shaped or unshaped) or prior to be plotted

Keyword arguments:
  * `vsel = collect(1:5)`: indices or parameter names of the parameters to be plotted. By default (up to) the first five parameters are plotted.
  * `bins::Union{Integer, Tuple{Union{Integer, AbstractRange}}, NamedTuple} = 200`: Number of bins or bin edges
  * `mean::Bool = false`: Indicate mean value, calculated via  `bat_stats().mean`, in all plots (currently only for samples)
  * `std::Bool = false`: Indicate the standard deviation of the mean calculated from `bat_stats().cov` in all plots (currently only for samples)
  * `globalmode::Bool = false`: Indicate global mode, calculated via `bat_stats().mode`, in all plots (currently only for samples)
  * `marginalmode::Bool = false`: Indicate local mode(s), i.e. the center of the highest histogram bin(s), in all plots (currently only for samples)
  *  `diagonal = Dict()`: The dictionary can contain the seriestypes and plot options for 1D distributions explained above to modify the plots of the 1D marginal distributions on the diagonal of the grid. Nested Dictionaries are possible to modify the styles of the estimators as described above
  * `lower = Dict()` and `upper = Dict()`:  The dictionaries can contain the seriestypes and plot options for 2D distributions explained above to modify the 2D plots in the lower and upper triangles of the grid. Nested Dictionaries are possible to modify the styles of the estimators as described above
  * `vsel_label = []`: parameter label as strings, `LatexStrings` are possible
