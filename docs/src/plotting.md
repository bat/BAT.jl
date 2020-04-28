# Plot Recipes

## 1D plots of samples
``` julia
plot(
    maybe_shaped_samples::DensitySampleVector,
    parsel::Union{Integer, Symbol};
    intervals = standard_confidence_vals,
    bins = 200,
    normalize = true,
    colors = standard_colors,
    interval_labels = [],
    mean = false,
    std = false,
    globalmode = false,
    localmode = true,
    filter = false,
    closed = :left
)
```
Required inputs:
  * `maybe_shaped_samples::DensitySampleVector`: BAT samples to be plotted (shaped or unshaped)
  * `parsel::Union{Integer, Symbol}`: index or parameter name as symybol of the parameter to be plotted

Keyword arguments:
  * `seriestype::Symbol = :smallest_intervals`: plot style
	available seriestypes:
	* `:smallest_intervals`, alias `:HDR` (default)
    * `:central_intervals`
    * `:histogram`, alias `:steppost`
    * `:stephist`
* `nbins::Integer = 200`: number of histogram bins

* `normalize::Bool = true`: normalize the histogram

* `intervals::Array{<:Real, 1} = [0.683, 0.955, 0.997]`: probability to be enclosed in the smallest/central intervals when using the corresponding seriestypes

* `colors::Array{Symbol, 1} = [:chartreuse2, :yellow, :red]`: colors of the smallest/central interval regions, in same order as the values in `intervals`

* `mean::Union{Dict, Bool} = true`: indicate mean value, calculated via `bat_stats().mean`.
The plot style of the mean can be customized using a `Dict`.  For `mean = true`, the default style is:  
`Dict("linestyle" => :solid, "linewidth" => 1, "linecolor" => :dimgrey, "alpha" => 1, "label" => "mean")`

* `std::Union{Dict, Bool} = true`: indicate the standard deviation of the mean, calculated from `bat_stats().cov`. The style of the standard deviation can be customized using a `Dict`.  For `std = true`, the default style is:   
`Dict("fillcolor" => :grey, "fillalpha" => 0.5, "label" => "std. dev.")`

* `globalmode::Union{Dict, Bool} = true`: indicate global mode, calculated via `bat_stats().mode`.  The style of the global mode can be passed as a `Dict`.  For `globalmode = true`, the default style is:  
`Dict("linestyle" => :dash, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "global mode")`

* `localmode::Union{Dict, Bool} = true`: indicates the localmode(s), i.e. the center of the highest histogram bin(s) . The style can be passed as a `Dict`. If `localmode = true`, the default style is:  
`Dict("linestyle" => :dot, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "local mode")`

* `filter::Bool = false`: if `true`, `BAT.drop_low_weight_samples()` is applied before plotting

* `closed = :left`: see [StatsBase.Histogram](https://juliastats.org/StatsBase.jl/stable/empirical/#StatsBase.fit-Tuple{Type{Histogram},Vararg{Any,N}%20where%20N})

Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can also be passed to the function.



## 2D plots of samples
``` julia
plot(
	maybe_shaped_samples::DensitySampleVector,
    parsel::Union{NTuple{2, Integer}, NTuple{2, Symbol}};
    intervals = standard_confidence_vals,
    colors = standard_colors,
    mean = false,
    std = false,
    globalmode = false,
    localmode = true,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = false,
    closed = :left
)
```
Required inputs:
  * `maybe_shaped_samples::DensitySampleVector`: BAT samples to be plotted (shaped or unshaped)
  * `parsel::Union{NTuple{2, Integer}, NTuple{2, Symbol}}`: indices or parameter names as symybols of the two parameters to be plotted

Keyword arguments:
  * `seriestype::Symbol = :smallest_intervals`: plot style
	available seriestypes:
	* `:smallest_intervals` (default)
	* `:smallest_intervals_contour`
	* `:smallest_intervals_contourf` (filled contours)
	* `:histogram`, alias `:hist`, alias `:histogram2d`
    * `:scatter`

* `nbins::Union{Integer, NTuple{2, Integer}} = 200`: number of histogram bins, use a `NTuple{2, Integer}` for differently binned x and y axes.

* `normalize::Bool = true`: normalize the histogram

* `intervals::Array{<:Real, 1} = [0.683, 0.955, 0.997]`  probability to be enclosed in the smallest intervals when using the corresponding seriestypes

* `colors::Array{Symbol, 1} = [:chartreuse2, :yellow, :red]` colors of the smallest interval regions, in same order as the values in `intervals`

* `mean::Union{Dict, Bool} = true`: indicate mean value, calculated via `bat_stats().mean`.
The plot style of the mean can be customized using a `Dict`.  For `mean = true`, the default style is:
`Dict("markercolor" => :black, "markersize" => 4, "markeralpha" => 1, "markershape" => :circle, "markerstrokealpha" => 1, "markerstrokecolor" => :black, "markerstrokestyle" => :solid, "markerstrokewidth" => 1, "label" => "mean")`

* `std::Union{Bool} = true`: indicate the standard deviation of the mean as errorbars, calculated from `bat_stats().cov`. The style of the errorbars can be customized using the `markerstroke...` options in `mean`.

* `globalmode::Union{Dict, Bool} = true`: indicate global mode, calculated via `bat_stats().mode`.  The style of the global mode can be passed as a `Dict`.  For `globalmode = true`, the default style is:
`Dict("linestyle" => :dash, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "global mode")`

* `localmode::Union{Dict, Bool} = true`: indicates the localmode(s), i.e. the center of the highest histogram bin(s) . The style can be passed as a `Dict`. If `localmode = true`, the default style is:
`Dict("linestyle" => :dot, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "local mode")`

Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can also be passed to the function.
