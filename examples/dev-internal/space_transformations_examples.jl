using BAT
using AffineMaps
using AutoDiffOperators
using Plots
using ValueShapes

context = BATContext(ad = ADModule(:ForwardDiff))

posterior = BAT.example_posterior()

target, trafo = BAT.transform_and_unshape(PriorToGaussian(), posterior, context)

s = BAT.cholesky(BAT._approx_cov(target, totalndof(varshape(target)))).L
f = BAT.CustomTransform(Mul(s))



propcov_result = BAT.bat_sample_impl(posterior, 
                                     MCMCSampling(adaptive_transform = f, nsteps = 4 * 100000), 
                                     context
)
propcov_samples = my_result.result
plot(propcov_samples)


ram_result = BAT.bat_sample_impl(posterior, 
                                 MCMCSampling(adaptive_transform = f, tuning = RAMTuning(), nsteps = 4 * 100000), 
                                 context
)
ram_samples = ram_result.result
plot(ram_samples)
