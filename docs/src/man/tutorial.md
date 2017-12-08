# Tutorial

## Starting with a Normal Distribution

First we need to set up the environment:
```julia
julia> using BAT
```

For our example we need additional packages:
```julia
julia> using Distributions, PDMats, StatsBase
```

Now we can define a Multi-variate Normal Distribution:
```julia
julia> mvec = [-0.3, 0.3]
julia> Σ = PDMat([1.0 1.5; 1.5 4.0])
julia> density = MvDistDensity(MvNormal(mvec, Σ))
```

Here we will use the `MetroplisHastings`-algorithm:
```julia
julia> algorithm = MetropolisHastings()
```
We want our samples to be in the following bounds with reflective property:
```julia
julia> bounds = HyperRectBounds([-5, -8], [5, 8], reflective_bounds)
```

To separate general MCMC specifications from specific MCMC settings and to hold
this information collectively the `MCMCSpec` type was introduced:
```julia
julia> spec = MCMCSpec(algorithm, density, bounds)
```

At last we define the total number of Markov-Chains and how many samples we want
for every chain:
```julia
julia> nsamples_per_chain = 2000
julia> nchains = 4
```

Then, we draw the samples with `rand`:
```julia
julia> samples, sampleids, stats = rand(
                        spec,
                        nsamples_per_chain,
                        nchains,
                        max_time = Inf,
                        granularity = 1
                        )               
```

We can check e.g. the sample covariance matrix:
```julia
julia> cov(samples.params, FrequencyWeights(samples.weight), 2; corrected=true)
2×2 Array{Float64,2}:
 0.971406  1.45419
 1.45419   3.97268
```
Indeed, the estimates are close to the true values of `Σ = [1.0 1.5; 1.5 4.0]`
which we used to specify the Normal Distribution.
