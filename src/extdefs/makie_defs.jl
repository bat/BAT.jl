# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct BATMakieVisualization <: BATVisBackend
        recipes::NamedTuple
        vsel::Vector{Integer}
        N_max::Integer
        n_batch::Integer
        triagonal_config::NamedTuple
        diagonal_config::NamedTuple
end
export BATMakieVisualization

function BATMakieVisualization()
        recipes = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D)
        vsel = [1, 2, 3] # Figure out default values. Pass samples to determine number of parameters?
        N_max = 3 # TODO: Can cause errors when the number of dimensions in the data is smaller than N_max. Figure out a way to make safe.
        n_batch = 10

        triagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=(100, 100),
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(2), 0:3),
                filter=false,
                colormap=:inferno,
                color=RGB(0.898, 0.361, 0.188),
                color_stats=:deepskyblue,
                alpha=1.0,
                rev=false,
                threshold=nothing,
                markersize=2.0,
                edge=false,
                strokecolor=RGB(0.741, 0.518, 0.02),
                strokewidth=1.0,
                strokestyle_stats=:solid,
                strokewidth_stats=2.0,
                color_mean=:white,
                strokestyle_mean=:dot,
                strokewidth_mean=2.0,
                color_ebars=:blue,
                whiskerwidth=10
        )

        diagonal_config = (
                weights=nothing,
                nsigma=1.0,
                nbins=100,
                closed=:left,
                normalization=:pdf,
                levels=cdf.(Chi(1), 0:3),
                filter=false,
                color=RGB(0.898, 0.361, 0.188),
                color_stats=:deepskyblue,
                colormap=:inferno,
                alpha=1.0,
                filled=true,
                edge=false,
                strokecolor=:orange,
                strokewidth=1.0,
                strokestyle_stats=:solid,
                strokewidth_stats=2.0,
                strokestyle_mean=:dot,
                strokewidth_mean=2.0,
                y_ebars=0.0,
                color_ebars=:blue,
                whiskerwidth=10,
                filled_pdf=true,
                npoints_pdf=300,
                rev=false
        )

        vis_config = BATMakieVisualization(
                recipes,
                vsel,
                N_max,
                n_batch,
                triagonal_config,
                diagonal_config,
        )
        return vis_config
end



#function init_visualizer(vis::BATVisualizer{BATMakieVisualization})
#return nothing
#end




function BATMakieVisualizer end
export BATMakieVisualizer

function init_visualizer!! end
export init_visualizer!!

function update_visualizer!! end
export update_visualizer!!


function bat_makie_plot end
export bat_makie_plot

function bat_makie_plot! end
export bat_makie_plot!


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


function cov2d end
export cov2d

function std1d end
export std1d

function std2d end
export std2d

function mean1d end
export mean1d

function mean2d end
export mean2d

function errorbars1d end
export errorbars1d

function errorbars2d end
export errorbars2d

function pdf1d end
export pdf1d


function bat_theme end
export bat_theme

function bat_theme_dark end
export bat_theme_dark
