# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ModeAsDefined <: AbstractModeEstimator

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.

Constructors:

* ```$(FUNCTIONNAME)()```        
"""
struct ModeAsDefined <: AbstractModeEstimator end
export ModeAsDefined


function bat_findmode_impl(target::AnySampleable, algorithm::ModeAsDefined)
    (result = StatsBase.mode(target),)
end

function bat_findmode_impl(target::Distribution, algorithm::ModeAsDefined)
    (result = varshape(target)(StatsBase.mode(unshaped(target))),)
end

function bat_findmode_impl(target::DistributionDensity, algorithm::ModeAsDefined)
    bat_findmode_impl(parent(target), algorithm)
end



"""
    MaxDensitySampleSearch <: AbstractModeEstimator

Constructors:

    MaxDensitySampleSearch()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct MaxDensitySampleSearch <: AbstractModeEstimator end
export MaxDensitySampleSearch


function bat_findmode_impl(target::DensitySampleVector, algorithm::MaxDensitySampleSearch)
    v, i = _get_mode(target)
    (result = v, mode_idx = i)
end
