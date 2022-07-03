# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, StatsBase, IntervalSets

import NestedSamplers
import UltraNest

mvec = [-0.3, 0.3]
cmat = [1.0 1.5; 1.5 4.0]
mv_dist = MvNormal(mvec, cmat)

likelihood1 = logfuncdensity(logdensityof(BAT.DistMeasure(mv_dist)))
likelihood2 = logfuncdensity(params -> begin
    logpdf(mv_dist, [params.x, params.y])
end)
likelihoods = [likelihood1, likelihood2]


prior1 = product_distribution(Uniform.([-5, -8], [5, 8]))
prior2 = NamedTupleDist(
    x = Uniform(-5,8),
    y = Uniform(5, 8),
)
priors = [prior1, prior2]

trafos = [DoNotTransform(), PriorToUniform()]

# run all samplers a shaped and an unshaped posterior with different transformations
for i in eachindex(priors), trafo in trafos
    prior = priors[i]
    likelihood = likelihoods[i]
    posterior = PosteriorMeasure(likelihood, prior)
    
    mh = MCMCSampling(mcalg = MetropolisHastings(), trafo = trafo, nchains = 2, nsteps = 10^2)
    sobol = SobolSampler(nsamples=10^2, trafo=trafo)
    nested_ellipsoidal = EllipsoidalNestedSampling() # only works with PriorToUniform()
    nested_reactive = ReactiveNestedSampling()

    algorithms = [nested_ellipsoidal, nested_reactive, mh, sobol]
    for algorithm in algorithms

        samples = bat_sample(posterior, algorithm)

        @test valshape(samples.result.v[1]) <= varshape(prior)
        @test valshape(samples.result_trafo.v[1]) <= ArrayShape{Real, 1}((vardof(prior),))
    end

end
