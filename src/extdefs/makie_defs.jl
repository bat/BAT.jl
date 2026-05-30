
struct BATMakieVisualizer <: BATVisBackend
        recipes::NamedTuple
        vsel::Vector{Integer}
        N_max::Integer
        n_batch::Integer
        graph::Any
        gridspec::Any
        ui_controls::NamedTuple
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

function hist1d end
export hist1d

function hist2d end
export hist2d

function quantilehist1d end
export quantilehist1d

function quantilehist2d end
export quantilehist2d

function hexbin2d end
export hexbin2d

function kde1d end
export kde1d

function kde2d end
export kde2d

function quantilekde1d end
export kde1d

function quantilekde2d end
export kde2d

function scatter2d end
export scatter2d

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

