# This file is a part of BAT.jl, licensed under the MIT License (MIT).

let
    posterior = BAT.example_posterior()

    dummy_samples = bat_sample(BAT._bat_determ_rng(), posterior.prior, IIDSampling(nsamples=10)).result
    precompile(bat_eff_sample_size, map(typeof, (dummy_samples,)))
    precompile(show, map(typeof, (dummy_samples,)))

    precompile(EvaluatedMeasure, map(typeof, (posterior, dummy_samples)))

    for mcalg in (RandomWalk(), HamiltonianMC())
        precompile(bat_sample, map(typeof, (posterior, TransformedMCMC(mcalg = mcalg))))
    end
end
