
struct BATMakieVisualizer <: BATVisBackend
        recipes::NamedTuple
        vsel::Vector{Integer}
        N_max::Integer
        n_batch::Integer
        triagonal_config::NamedTuple
        diagonal_config::NamedTuple
end
export BATMakieVisualizer

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

