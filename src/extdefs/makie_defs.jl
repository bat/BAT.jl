# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct BATMakieVisualization <: BATVisBackend
        recipes::NamedTuple
        vsel::Vector{Integer}
        N_max::Integer
        n_batch::Integer
        # Backpressure watermark: once this many samples are buffered and not yet
        # flushed, sampling threads block (in update_visualizer_impl!) until the
        # listener catches up. Bounds how far the display can lag behind the true
        # sampler state without throttling every single sample -- sampling still
        # runs in free bursts up to this many samples ahead. When adaptive_batching
        # is on, this (like n_batch) is only the *starting* value -- both grow
        # together at runtime, in the same ratio to one another as configured here.
        max_buffered::Integer
        # When true (the default), the effective flush threshold grows by
        # batch_growth_rate after every flush (geometric growth, the same
        # amortized-doubling trick as dynamic array growth), so the expensive
        # per-flush recompute (histogram/KDE fits over the *entire* accumulated
        # sample set) happens on a shrinking fraction of the run as it progresses
        # -- fixing an otherwise-quadratic total cost over a full run. When false,
        # n_batch/max_buffered stay exactly as configured for the whole run --
        # more frequent, uniformly-sized updates, at the cost of more total
        # recompute work, for users who want maximum live-update resolution.
        adaptive_batching::Bool
        batch_growth_rate::Real
        poll_interval::Real
        dark::Bool
        triagonal_config::NamedTuple
        diagonal_config::NamedTuple

        # N_max fixes the grid size for the life of the visualizer; vsel selects
        # which N_max (or fewer) variables are shown and may later be changed at
        # runtime (via a not-yet-implemented UI widget), but can never select more
        # variables than there are grid slots for.
        function BATMakieVisualization(recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching, batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config)
                if length(vsel) > N_max
                        @warn "BATMakieVisualization: vsel $vsel has more entries than N_max=$N_max; truncating to $(vsel[1:N_max])."
                        vsel = vsel[1:N_max]
                end
                if batch_growth_rate <= 1
                        throw(ArgumentError("batch_growth_rate must be > 1 (got $batch_growth_rate); it multiplies the effective batch threshold after every flush"))
                end
                new(recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching, batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config)
        end
end
export BATMakieVisualization

function BATMakieVisualization(; dark::Bool=false, max_buffered::Integer=4 * 50, adaptive_batching::Bool=true, batch_growth_rate::Real=1.2)
        recipes = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D)
        vsel = [1, 2, 3] # Default vsel; truncated in `init_visualizer!` if the posterior has fewer free parameters.
        N_max = 3 # Grid size; cells beyond the (possibly truncated) vsel are simply left dead/empty.
        n_batch = 50 # Flush the buffered samples into the plot once this many have accumulated.
        poll_interval = 0.1 # Seconds between checks of whether a new batch is ready to flush.

        triagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=(100, 100),
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(2), 0:3),
                filter=false,
                colormap=:inferno,
                alpha=1.0,
                rev=false,
                threshold=nothing,
                markersize=2.0
        )

        diagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=100,
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(1), 0:3),
                filter=false,
                colormap=:inferno,
                alpha=1.0,
                y_ebars=0.0,
                filled_pdf=true,
                npoints_pdf=300,
                rev=false
        )

        vis_config = BATMakieVisualization(
                recipes,
                vsel,
                N_max,
                n_batch,
                max_buffered,
                adaptive_batching,
                batch_growth_rate,
                poll_interval,
                dark,
                triagonal_config,
                diagonal_config,
        )
        return vis_config
end



function bat_makie_plot end
export bat_makie_plot


abstract type BATMakieRecipe end
export BATMakieRecipe


struct Hist1D <: BATMakieRecipe end
export Hist1D

struct Hist2D <: BATMakieRecipe end
export Hist2D

struct QuantileHist1D <: BATMakieRecipe end
export QuantileHist1D

struct QuantileHist2D <: BATMakieRecipe end
export QuantileHist2D

struct Hexbin2D <: BATMakieRecipe end
export Hexbin2D


struct Scatter2D <: BATMakieRecipe end
export Scatter2D

# Colors points by MCMC chain instead of a single flat color -- only
# meaningful (and only ever offered in the UI) when the plotted samples
# actually carry chain identity, i.e. `eltype(samples.info) <: SampleID` with
# a `chainid` field (true for MCMCSampleID/AHMCSampleID, false for e.g.
# importance sampling or MGVI, whose samples have no chain concept at all).
struct ChainScatter2D <: BATMakieRecipe end
export ChainScatter2D


struct KDE1D <: BATMakieRecipe end
export KDE1D

struct KDE2D <: BATMakieRecipe end
export KDE2D

struct QuantileKDE1D <: BATMakieRecipe end
export QuantileKDE1D

struct QuantileKDE2D <: BATMakieRecipe end
export QuantileKDE2D


struct Cov2D <: BATMakieRecipe end
export Cov2D

struct Std1D <: BATMakieRecipe end
export Std1D

struct Std2D <: BATMakieRecipe end
export Std2D

struct Mean1D <: BATMakieRecipe end
export Mean1D

struct Mean2D <: BATMakieRecipe end
export Mean2D

struct Errorbars1D <: BATMakieRecipe end
export Errorbars1D

struct Errorbars2D <: BATMakieRecipe end
export Errorbars2D

struct PDF1D <: BATMakieRecipe end
export PDF1D


function bat_theme end
export bat_theme

function bat_theme_dark end
export bat_theme_dark
