using BAT
using AdvancedHMC
using AffineMaps
using AutoDiffOperators
using ValueShapes



context = BATContext(ad = ADModule(:ForwardDiff))

posterior = BAT.example_posterior()

target, trafo = BAT.transform_and_unshape(PriorToGaussian(), posterior, context)

s = BAT.cholesky(BAT._approx_cov(target, totalndof(varshape(target)))).L
f = BAT.CustomTransform(Mul(s))


# Metropolis-Hastings MC Sampling

propcov_result = BAT.bat_sample_impl(posterior, 
                                     MCMCSampling(adaptive_transform = f), 
                                     context
)
propcov_samples = propcov_result.result

using Plots
plot(propcov_samples)


ram_result = BAT.bat_sample_impl(posterior, 
                                 MCMCSampling(adaptive_transform = f, trafo_tuning = RAMTuning()), 
                                 context
)
ram_samples = ram_result.result
plot(ram_samples)

# Advanced Hamiltonian MC Sampling

hmc_result = BAT.bat_sample_impl(posterior,
                                 MCMCSampling(adaptive_transform = f, proposal = HamiltonianMC(), trafo_tuning = StanHMCTuning()),
                                 context
)
hmc_samples = hmc_result.result
plot(hmc_samples)
