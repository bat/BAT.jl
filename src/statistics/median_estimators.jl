# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct SampleMedianEstimator <: AbstractMedianEstimator

Get median values from samples using standard Julia statistics functions.

Constructors:

* ```$(FUNCTIONNAME)()```        
"""
struct SampleMedianEstimator <: AbstractMedianEstimator end
export SampleMedianEstimator


function bat_findmedian_impl(samples::DensitySampleVector, ::SampleMedianEstimator, ::BATContext)
    median_params = median(samples)
    (result = median_params,)
end
