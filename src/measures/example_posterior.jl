# This file is a part of BAT.jl, licensed under the MIT License (MIT).


example_stable_rng() = StableRNGs.StableRNG(0x4cf83495c736cac2)


function example_prior()
    return NamedTupleDist(
        a = Exponential(),
        b = [4.2, 3.3],
        c = Normal(1, 3),
        d = [Weibull(), Weibull()],
        e = Beta(),
        f = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    )
end

function example_prior_with_dirichlet()
    merge(example_prior(), (g = Dirichlet([1.2, 2.4, 3.6]),))
end



function example_likelihood(prior::Distribution, rng::AbstractRNG)
    n = totalndof(varshape(prior))
    A = randn(rng, n, n)
    likelihood = logfuncdensity(logdensityof(varshape(prior)(MvNormal(A * A'))))
    # ....
    return Likelihood(data, model)
end


function example_posterior(rng::AbstractRNG = example_stable_rng(), prior::Distribution = example_prior())
    prior = example_prior()
    likelihood = example_likelihood(prior, rng)
    lbqintegral(likelihood, prior)
end

function example_posterior_with_dirichlet(rng::AbstractRNG = example_stable_rng())
    return example_posterior(rng, example_prior_with_dirichlet())
end


function old_example_posterior()
    rng = example_stable_rng()
    n = totalndof(varshape(prior))
    A = randn(rng, n, n)
    likelihood = logfuncdensity(logdensityof(varshape(prior)(MvNormal(A * A'))))
    lbqintegral(likelihood, prior)
end

function old_example_posterior_with_dirichlet()
    rng = example_stable_rng()
    prior = merge(BAT.example_posterior().prior.dist, (g = Dirichlet([1.2, 2.4, 3.6]),))
    n = totalndof(varshape(prior))
    A = randn(rng, n, n)
    likelihood = logfuncdensity(logdensityof(varshape(prior)(MvNormal(A * A'))))
    PosteriorMeasure(likelihood, prior)
end
