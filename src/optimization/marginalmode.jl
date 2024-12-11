# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct BinnedMarginalModes <: AbstractModeEstimator

*Experimental feature, not part of stable public API.*

Bin data to estimate modes.

Constructor: `$(FUNCTIONNAME)(; fields...)`

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct BinnedMode{BA} <: AbstractModeEstimator
    binning::BA = FreedmanDiaconisBinning()
end


function bat_marginalmode_impl(samples::DensitySampleVector, algorithm::BinnedMode, context::BATContext)
    shape = varshape(samples)
    flat_samples = flatview(unshaped.(samples.v))
    n_params = size(flat_samples)[1]
    nt_samples = ntuple(i -> flat_samples[i,:], n_params)
    marginalmode_params = Vector{Float64}()

    bin_edges = _get_binedges(nt_samples, algorithm.binning, context)

    for param in Base.OneTo(n_params)
        # ToDo: Forward binning edges instead of just number of bins:
        marginalmode_param = find_marginalmodes(MarginalDist(samples, param, bins = length(bin_edges[param])-1))

        if length(marginalmode_param[1]) > 1
            @warn "More than one bin with the same weight is found. Returned the first one"
        end
        push!(marginalmode_params, marginalmode_param[1][1])
    end
    (result = shape(marginalmode_params),)
end
