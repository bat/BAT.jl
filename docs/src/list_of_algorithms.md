
# List of algorithms 

BAT.jl offers performant implementations of mutliple algorithms for sampling, integration and optimization.

## Sampling algorithms

- [IIDSampling](https://bat.github.io/BAT.jl/dev/stable_api/#BAT.IIDSampling)
	```
	IIDSampling(nsamples=10^5)
	```

- [Metropolis-Hastings]([https://bat.github.io/BAT.jl/dev/stable_api/#BAT.MetropolisHastings](https://bat.github.io/BAT.jl/dev/stable_api/#BAT.MetropolisHastings))
	```
	MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^6, nchains = 4)
	```

-   [Hamiltonian MC](https://bat.github.io/BAT.jl/dev/stable_api/#BAT.HamiltonianMC)
	```
	MCMCSampling(mcalg = HamiltonianMC())
	```

-   [Reactive Nested Sampling](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.ReactiveNestedSampling)
	```
	ReactiveNestedSampling()
	```

-   [Partitioned Sampling](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.PartitionedSampling)

-   [Sobol Sampler](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.SobolSampler)
	```
	SobolSampler(nsamples=10^5)
	```


-   [Grid Sampler](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.GridSampler)
	```
	GridSampler(ppa=100)
	```
-   [Prior Importance Sampler](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.PriorImportanceSampler)
	```
	GridSampler(nsamples=10^5)
	```


## Integration algorithms:

-   [Adaptive Harmonic Mean Integration (AHMI)](https://bat.github.io/BAT.jl/dev/stable_api/#BAT.AHMIntegration)

-   [Vegas Integration](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.VEGASIntegration)

-   [Cuhre Integration](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.CuhreIntegration)

-   [Divonne Integration](https://bat.github.io/BAT.jl/dev/experimental_api/#BAT.DivonneIntegration)

## Optimization algorithms:

-  [Nelder-Mead algorithm](https://bat.github.io/BAT.jl/dev/stable_api/#BAT.MaxDensityNelderMead)

-   [LBFGS](https://bat.github.io/BAT.jl/dev/stable_api/#BAT.MaxDensityLBFGS)
