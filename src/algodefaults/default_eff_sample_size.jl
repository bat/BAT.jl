# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    samples::AbstractVectorOfSimilarVectors{<:Real},
) = EffSampleSizeFromAC()

# Provenance-driven: autocorrelation ESS where an ordered sampling
# process is identifiable (uniform weights - the stored order is the
# process order - or per-sample MCMC ids), weight-degeneracy (Kish) ESS
# for nonuniformly weighted samples without process provenance:
function bat_default(
    ::typeof(bat_eff_sample_size),
    ::Val{:algorithm},
    samples::DensitySampleVector,
)
    W = samples.weight
    if isempty(W) || all(w -> w ≈ first(W), W) || eltype(samples.info) <: MCMCSampleID
        return EffSampleSizeFromAC()
    else
        return KishESS()
    end
end
