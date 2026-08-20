# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    samples::AbstractVectorOfSimilarVectors{<:Real},
) = EffSampleSizeFromAC()

# Provenance-driven: autocorrelation ESS where an ordered sampling
# process is identifiable (usable per-sample MCMC ids, or uniform weights
# - then the stored order is the process order), weight-degeneracy (Kish)
# ESS for nonuniformly weighted samples without process provenance:
function bat_default(
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    samples::DensitySampleVector,
)
    W = samples.weight
    if isempty(W) || all(w -> w ≈ first(W), W) || _has_process_provenance(samples)
        return EffSampleSizeFromAC()
    else
        return KishESS()
    end
end
