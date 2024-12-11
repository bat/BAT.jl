# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct PredefinedOptimum <: AbstractOptimizer

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.

Constructors:

* ```$(FUNCTIONNAME)()```        
"""
struct PredefinedOptimum <: AbstractOptimizer end
export PredefinedOptimum


function bat_findmode_impl(target::AnySampleable, algorithm::PredefinedOptimum, context::BATContext)
    (result = StatsBase.mode(target),)
end

function bat_findmode_impl(target::Distribution, algorithm::PredefinedOptimum, context::BATContext)
    (result = varshape(target)(StatsBase.mode(unshaped(target))),)
end

function bat_findmode_impl(target::BATDistMeasure, algorithm::PredefinedOptimum, context::BATContext)
    bat_findmode_impl(Distribution(target), algorithm, context)
end



"""
    EmpiricalOptimum <: AbstractOptimizer

Constructors:

    EmpiricalOptimum()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct EmpiricalOptimum <: AbstractOptimizer end
export EmpiricalOptimum


function bat_findmode_impl(target::DensitySampleVector, algorithm::EmpiricalOptimum, context::BATContext)
    v, i = _get_mode(target)
    (result = v, mode_idx = i)
end
