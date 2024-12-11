# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct PredefinedMode <: AbstractModeEstimator

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.

Constructors:

* ```$(FUNCTIONNAME)()```        
"""
struct PredefinedMode <: AbstractModeEstimator end
export PredefinedMode


function bat_findmode_impl(target::AnySampleable, algorithm::PredefinedMode, context::BATContext)
    (result = StatsBase.mode(target),)
end

function bat_findmode_impl(target::Distribution, algorithm::PredefinedMode, context::BATContext)
    (result = varshape(target)(StatsBase.mode(unshaped(target))),)
end

function bat_findmode_impl(target::BATDistMeasure, algorithm::PredefinedMode, context::BATContext)
    bat_findmode_impl(Distribution(target), algorithm, context)
end



"""
    EmpiricalMode <: AbstractModeEstimator

Constructors:

    EmpiricalMode()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct EmpiricalMode <: AbstractModeEstimator end
export EmpiricalMode


function bat_findmode_impl(target::DensitySampleVector, algorithm::EmpiricalMode, context::BATContext)
    v, i = _get_mode(target)
    (result = v, mode_idx = i)
end
