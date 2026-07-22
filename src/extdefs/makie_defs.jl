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
        # runs in free bursts up to this many samples ahead.
        max_buffered::Integer
        poll_interval::Real
        dark::Bool
        triagonal_config::NamedTuple
        diagonal_config::NamedTuple

        # N_max fixes the grid size for the life of the visualizer; vsel selects
        # which N_max (or fewer) variables are shown and may later be changed at
        # runtime (via a not-yet-implemented UI widget), but can never select more
        # variables than there are grid slots for.
        function BATMakieVisualization(recipes, vsel, N_max, n_batch, max_buffered, poll_interval, dark, triagonal_config, diagonal_config)
                if length(vsel) > N_max
                        @warn "BATMakieVisualization: vsel $vsel has more entries than N_max=$N_max; truncating to $(vsel[1:N_max])."
                        vsel = vsel[1:N_max]
                end
                new(recipes, vsel, N_max, n_batch, max_buffered, poll_interval, dark, triagonal_config, diagonal_config)
        end
end
export BATMakieVisualization

function BATMakieVisualization(; dark::Bool=false, max_buffered::Integer=4 * 50)
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
