# Plot Recipes
The plot recipes described below apply for plotting samples. Only the plotting of estimators (mean, standard deviation, global mode and local mode) is currently only supported for samples.

See the [tutorial](tutorial.md) for examples.

## Selecting parameters

Parameters are selected by name, `:a`, by element of an array-valued
parameter, `:(a[2])`, or by path in nested named tuples, `:(x.b.c)`.

They can also be selected by position, as an integer. Positions count the
free parameters in the order of the flat real vectors that `cov(samples)`
and `unshaped.(samples)` use: named tuple fields in order, nested fields
depth-first, arrays element by element in column-major order, and constant
parameters left out. Samples without names, like samples of flat vectors,
can only be selected by position, their parameters are labeled `v_1`,
`v_2`, ... (or `v` for scalar variates).

Histograms, intervals, means and standard deviations are weighted with the sample weights.

## 1D plots
``` julia
plot(
    samples::DensitySampleVector,
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
  * `samples::DensitySampleVector`: samples to be plotted
  * `vsel::Union{Integer, Symbol, Expr}`: position or name of the parameter to be plotted, see [Selecting parameters](@ref)

Keyword arguments:
  * `seriestype::Symbol = :smallest_intervals`: plot style        
	available seriestypes:
	* `:smallest_intervals` (default for samples), alias `:HDR`
    * `:central_intervals`
    * `:histogram`, alias `:steppost`
    * `:stephist`

* `bins::Union{Integer, AbstractRange} = 200`: number of histogram bins or bin edges.

* `intervals::Array{<:Real, 1} = [0.683, 0.955, 0.997]`: probability to be enclosed in the smallest/central intervals when using the corresponding seriestypes

* `colors::Array{Symbol, 1} = [:chartreuse2, :yellow, :red]`: colors of the smallest/central interval regions, in same order as the values in `intervals`

* `mean::Union{Dict, Bool} = true`: indicate mean value (currently only for samples), calculated via `mean(samples)`.
The plot style of the mean can be customized using a `Dict`.  For `mean = true`, the default style is:  
`Dict("linestyle" => :solid, "linewidth" => 1, "linecolor" => :dimgrey, "alpha" => 1, "label" => "mean")`

* `std::Union{Dict, Bool} = true`: indicate the standard deviation of the mean (currently only for samples), calculated from `cov(samples)`. The style of the standard deviation can be customized using a `Dict`.  For `std = true`, the default style is:   
`Dict("fillcolor" => :grey, "fillalpha" => 0.5, "label" => "std. dev.")`

* `globalmode::Union{Dict, Bool} = true`: indicate global mode (currently only for samples), calculated via `mode(samples)`.  The style of the global mode can be passed as a `Dict`.  For `globalmode = true`, the default style is:  
`Dict("linestyle" => :dash, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "global mode")`

* `marginalmode::Union{Dict, Bool} = true`: indicate the marginal mode(s), i.e. the center of the highest histogram bin(s) (currently only for samples). The style can be passed as a `Dict`. If `marginalmode = true`, the default style is:  
`Dict("linestyle" => :dot, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "marginal mode")`

* (only for samples) `filter::Bool = false`: if `true`, `BAT.drop_low_weight_samples()` is applied before plotting

* `closed = :left`: see [StatsBase.Histogram](https://juliastats.org/StatsBase.jl/stable/empirical/#StatsBase.fit-Tuple{Type{Histogram},Vararg{Any,N}%20where%20N})

Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can also be passed to the function.



## 2D plots
``` julia
plot(
    samples::DensitySampleVector,
    vsel::Union{NTuple{2, Integer}, NTuple{2, Union{Symbol, Expr}}};
    intervals = BAT.default_credibilities,
	interval_labels = [],
    colors = BAT.default_colors,
	bins = 200,
    mean = false,
    std = false,
    globalmode = false,
    marginalmode = true,
	smoothing = 0,
    diagonal = Dict(),
    upper = Dict(),
    right = Dict(),
    filter = false,
    closed = :left
)
```
Required inputs:
  * `samples::DensitySampleVector`: samples to be plotted
  * `vsel::Union{NTuple{2, Integer}, NTuple{2, Union{Symbol, Expr}}}`: positions or names of the two parameters to be plotted

Keyword arguments:
  * `seriestype::Symbol = :smallest_intervals`: plot style  
	available seriestypes:
	* `:smallest_intervals` (default for samples)
	* `:smallest_intervals_contour`
	* `:smallest_intervals_contourf`: filled contours)
	* `:histogram`, alias `:hist`, alias `:histogram2d`
    * `:scatter`


* `bins::Union{Integer, NTuple{2, Integer}, NTuple{2, AbstractRange}} = 200`: number of histogram bins or bin edges. Use a `NTuple{2, Union{Integer, AbstractRange}}` to specify bins/edges of x and y axes separately.

* `intervals::Array{<:Real, 1} = [0.683, 0.955, 0.997]`  probability to be enclosed in the smallest intervals when using the corresponding seriestypes

* `interval_labels = []`  label of the intervals as strings, `LatexStrings` are possible.

* `colors::Array{Symbol, 1} = [:chartreuse2, :yellow, :red]` colors of the smallest interval regions, in same order as the values in `intervals`

* `mean::Union{Dict, Bool} = true`: indicate mean value (currently only for samples) , calculated via `mean(samples)`.
The plot style of the mean can be customized using a `Dict`.  For `mean = true`, the default style is:
`Dict("markercolor" => :black, "markersize" => 4, "markeralpha" => 1, "markershape" => :circle, "markerstrokealpha" => 1, "markerstrokecolor" => :black, "markerstrokestyle" => :solid, "markerstrokewidth" => 1, "label" => "mean")`

* `std::Union{Bool} = true`: indicate the standard deviation of the mean as errorbars (currently only for samples), calculated from `cov(samples)`. The style of the errorbars can be customized using the `markerstroke...` options in `mean`.

* `globalmode::Union{Dict, Bool} = true`: indicate global mode (currently only for samples), calculated via `mode(samples)`.  The style of the global mode can be passed as a `Dict`.  For `globalmode = true`, the default style is:
`Dict("linestyle" => :dash, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "global mode")`

* `marginalmode::Union{Dict, Bool} = true`: indicate the marginalmode(s), i.e. the center of the highest histogram bin(s) (currently only for samples). The style can be passed as a `Dict`. If `marginalmode = true`, the default style is:
`Dict("linestyle" => :dot, "linewidth" => 1, "linecolor" => :black, "alpha" => 1, "label" => "marginal mode")`

* `smoothing = 0`: When plotting contours, a Gaussian filtering can be applied for smoothing the contour lines. The keyword `smoothing` accepts  positive real number (or a tuple of two positive real numbers), specifying the standard deviation of the Gaussian kernel (for each dimension) of the filtering.

*  `diagonal = Dict()`: Used only for the seriestype `:marginal`. The dictionary can contain the seriestypes and plot options for 2D distributions explained above to modify the 2D plot of the marginal plot. Nested Dictionaries are possible to modify the styles of the estimators as described above

* `upper = Dict()` and `right = Dict()`:  Used only for the seriestype `:marginal`. The dictionaries can contain the seriestypes and plot options for 1D distributions explained above to modify the upper and right plots of the 1D marginal distributions. Nested Dictionaries are possible to modify the styles of the estimators as described above

* `filter::Bool = false`: if `true`, `BAT.drop_low_weight_samples()` is applied before plotting

* `closed = :left`: see [StatsBase.Histogram](https://juliastats.org/StatsBase.jl/stable/empirical/#StatsBase.fit-Tuple{Type{Histogram},Vararg{Any,N}%20where%20N})

Keyword arguments for [attributes supported by *Plots.jl*](https://docs.juliaplots.org/latest/attributes/#magic-arguments) can also be passed to the function.


## 1D & 2D overview plots
``` julia
plot(
	samples::DensitySampleVector;
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
  * `samples::DensitySampleVector`: samples to be plotted

Keyword arguments:
  * `vsel = collect(1:5)`: positions or names of the parameters to be plotted. By default (up to) the first five parameters are plotted.
  * `bins::Union{Integer, Tuple{Union{Integer, AbstractRange}}, NamedTuple} = 200`: Number of bins or bin edges
  * `mean::Bool = false`: Indicate mean value, calculated via `mean(samples)`, in all plots (currently only for samples)
  * `std::Bool = false`: Indicate the standard deviation of the mean calculated from `cov(samples)` in all plots (currently only for samples)
  * `globalmode::Bool = false`: Indicate global mode, calculated via `mode(samples)`, in all plots (currently only for samples)
  * `marginalmode::Bool = false`: Indicate local mode(s), i.e. the center of the highest histogram bin(s), in all plots (currently only for samples)
  *  `diagonal = Dict()`: The dictionary can contain the seriestypes and plot options for 1D distributions explained above to modify the plots of the 1D marginal distributions on the diagonal of the grid. Nested Dictionaries are possible to modify the styles of the estimators as described above
  * `lower = Dict()` and `upper = Dict()`:  The dictionaries can contain the seriestypes and plot options for 2D distributions explained above to modify the 2D plots in the lower and upper triangles of the grid. Nested Dictionaries are possible to modify the styles of the estimators as described above
  * `vsel_label = []`: parameter label as strings, `LatexStrings` are possible


## MCMC diagnostics
``` julia
plot(
    BAT.MCMCDiagnostics(samples::DensitySampleVector);
    vsel = collect(1:totalndof(varshape(samples))),
    chains = collect(eachindex(unique(samples.info.chainid))),
    diagnostics = [:histogram, :kde, :trace, :acf],
    histogram = Dict(),
    kde = Dict(),
    trace = Dict(),
    acf = Dict(),
    description = true
)
```
Plot diagnostics of MCMC samples for each selected parameter and chain. The samples must carry the chain IDs of MCMC output in `samples.info.chainid`.

Keyword arguments:
  * `vsel`: positions of the parameters to be plotted, by default all free parameters
  * `chains`: chains to be plotted, as positions in the sorted list of chain IDs, by default all chains
  * `diagnostics`: diagnostics to be plotted:
    * `:histogram`: histogram of the samples, see 1D plots above
    * `:kde`: kernel density estimate, using [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl)
    * `:trace`: trace of the parameter values
    * `:acf`: autocorrelation function, using [`StatsBase.autocor`](https://juliastats.org/StatsBase.jl/stable/signalcorr/#StatsBase.autocor)
  * `histogram`, `kde`, `trace`, `acf`: options for each diagnostic as a `Dict`, including attributes supported by Plots.jl. `histogram` takes the options of 1D plots. `kde` also takes `"npoints"` (default 2048), `"boundary"`, `"kernel"` (default `Normal`) and `"bandwidth"`, `acf` also takes `"lags"` and `"demean"`.
  * `description::Bool = true`: show the chain, the parameter and the number of samples in the first column

The histogram and the kernel density estimate use the sample weights. The trace and the autocorrelation function follow the samples in the order they are stored and ignore the weights. With the default `RepetitionWeighting`, MCMC output stores a sample that the chain repeats only once, with the number of repetitions as its weight. The trace and the autocorrelation function then show the sequence of distinct samples.
