using BAT
using BAT.MeasureBase
using AffineMaps
using ChangesOfVariables
using BAT.LinearAlgebra
using BAT.Distributions
using BAT.InverseFunctions
import BAT: TransformedMCMCIterator, TransformedAdaptiveMHTuning, TransformedRAMTuner, TransformedMHProposal, TransformedNoMCMCTempering, transformed_mcmc_step!!, TransformedMCMCSampleID
using BAT.Random123

import BAT: mcmc_iterate!, transformed_mcmc_iterate!, TransformedMCMCSampling

#ENV["JULIA_DEBUG"] = "BAT"

rng = Philox4x()

posterior = BAT.example_posterior()

my_result = @time BAT.bat_sample_impl(rng, posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000))
my_samples = my_result.result

mh_result = @time BAT.bat_sample_impl(rng, posterior, TransformedMCMCSampling(tuning_alg=TransformedAdaptiveMHTuning(), pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000))

(;chain, tuner) = BAT.g_state


using Plots
plot(my_samples)

r_mh = @time BAT.bat_sample_impl(rng, posterior, MCMCSampling( nchains=4, nsteps=4*100000, store_burnin=true) )

r_hmc = @time BAT.bat_sample_impl(rng, posterior, MCMCSampling(mcalg=HamiltonianMC(), nchains=4, nsteps=4*20000) )
 
plot(bat_sample(posterior).result)

using BAT.Distributions
using BAT.ValueShapes
prior2 = NamedTupleDist(ShapedAsNT,
    b = [4.2, 3.3],
    a = Exponential(1.0),
    c = Normal(1.0,3.0),
    d = product_distribution(Weibull.(ones(2),1)),
    e = Beta(1.0, 1.0),
    f = MvNormal([0.3,-2.9],Matrix([1.7 0.5;0.5 2.3]))
    )

posterior.likelihood.density._log_f(rand(posterior.prior))

posterior.likelihood.density._log_f(rand(prior2))

posterior2 = PosteriorDensity(BAT.logfuncdensity(posterior.likelihood.density._log_f), prior2)


@profview r_ram2 = @time BAT.bat_sample_impl(rng, posterior2, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000))

@profview r_mh2 = @time BAT.bat_sample_impl(rng, posterior2, MCMCSampling( nchains=4, nsteps=4*100000, store_burnin=true) )

r_hmc2 = @time BAT.bat_sample_impl(rng, posterior2, MCMCSampling(mcalg=HamiltonianMC(), nchains=4, nsteps=4*20000) )

