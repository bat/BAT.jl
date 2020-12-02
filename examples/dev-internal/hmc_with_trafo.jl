include(joinpath(dirname(dirname(@__DIR__)), "docs", "src", "tutorial_lit.jl"))

samples_mh = bat_sample(posterior, 10^5, MCMCSampling(sampler = MetropolisHastings())).result

posterior_is = bat_transform(PriorToGaussian(), posterior, PriorSubstitution()).result
posterior_is2 = bat_transform(PriorToGaussian(), posterior, FullDensityTransform()).result

trafo_is = trafoof(posterior_is.likelihood)
trafo_is2 = trafoof(posterior_is2)

samples_is = bat_sample(posterior_is, 10^5, MCMCSampling(sampler = HamiltonianMC())).result
samples_is2 = bat_sample(posterior_is2, 10^5, MCMCSampling(sampler = HamiltonianMC())).result

samples = inv(trafo_is).(samples_is)
samples2 = inv(trafo_is2).(samples_is2)

plot(samples)

plot(-4:0.01:4, fit_function, samples)
plot!(
    normalize(hist, mode=:density),
    color=1, linewidth=2, fillalpha=0.0,
    st = :steps, fill=false, label = "Data",
    title = "Data, True Model and Best Fit"
)
plot!(-4:0.01:4, x -> fit_function(true_par_values, x), color=4, label = "Truth")

trafo_us_is = BAT.DistributionTransform(Uniform, trafoof(posterior_is.likelihood).target_dist)
samples_us = trafo_us_is.(samples_is)

bat_eff_sample_size(samples_mh)
bat_eff_sample_size(samples_is)
bat_eff_sample_size(samples_is2)
bat_eff_sample_size(samples)
bat_eff_sample_size(samples_us)

bat_integrate(samples_mh, AHMIntegration()).result
bat_integrate(samples_is, AHMIntegration()).result
bat_integrate(samples_is2, AHMIntegration()).result
bat_integrate(samples, AHMIntegration()).result
bat_integrate(samples_us, AHMIntegration()).result


mode_lbfgs = bat_findmode(posterior, MaxDensityLBFGS(init = ExplicitInit([mode(samples)]))).result

mode_lbfgs_is = bat_findmode(posterior_is, MaxDensityLBFGS(init = ExplicitInit([mode(samples_is)]))).result
mode_lbfgs_fromis = inv(trafo_is)(mode_lbfgs_is)

posterior_is_novc = trafoof(posterior_is.likelihood)(posterior; volcorr = Val(false))
mode_lbfgs_is_novc = bat_findmode(posterior_is_novc, MaxDensityLBFGS(init = ExplicitInit([mode(samples_is)]))).result
mode_lbfgs_fromis_novc = inv(trafo_is)(mode_lbfgs_is_novc)
