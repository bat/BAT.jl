# List of algorithms 

BAT.jl offers performant implementations of mutliple algorithms for sampling, integration and optimization.

## Sampling algorithms

All following sampling algorithms can be passed to [`bat_sample`](@ref):
```julia
samples = bat_sample(sampleable, sampling_algorithm).result
```

### IIDSampling
BAT.jl sampling algorithm type: [`IIDSampling`](@ref)
```julia
sampling_algorithm = IIDSampling(nsamples=10^5)
```


### Metropolis-Hastings
BAT.jl sampling algorithm type: [`MCMCSampling`](@ref)  
MCMC algorithm subtype: [`MetropolisHastings`](@ref)
```julia
sampling_algorithm = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^6, nchains = 4)
```



### Hamiltonian MC
BAT.jl sampling algorithm type: [`MCMCSampling`](@ref)  
MCMC algorithm subtype: [`HamiltonianMC`](@ref)
```julia
sampling_algorithm = MCMCSampling(mcalg = HamiltonianMC())
```

### Reactive Nested Sampling (experimental)
BAT.jl sampling algorithm type: [`ReactiveNestedSampling`](@ref)
```julia
sampling_algorithm = ReactiveNestedSampling()
```

### Ellipsoidal Nested Sampling (experimental)
BAT.jl sampling algorithm type: [`EllipsoidalNestedSampling`](@ref)
```julia
sampling_algorithm = EllipsoidalNestedSampling()
```

### Partitioned Sampling (experimental)
BAT.jl sampling algorithm type: `PartitionedSampling`, requires [PartitionedParallelSampling.jl](https://github.com/bat/PartitionedParallelSampling.jl).
```julia
sampling_algorithm = PartitionedParallelSampling.PartitionedSampling()
```


### Sobol Sampler
BAT.jl sampling algorithm type: [`SobolSampler`](@ref)
```julia
sampling_algorithm = SobolSampler(nsamples=10^5)
```


### Grid Sampler
BAT.jl sampling algorithm type: [`GridSampler`](@ref)
```julia
sampling_algorithm = GridSampler(ppa=100)
```


## Prior Importance Sampler
BAT.jl sampling algorithm type: [`PriorImportanceSampler`](@ref)
```julia
sampling_algorithm = GridSampler(nsamples=10^5)
```


## Integration algorithms:
All following integration algorithms can be passed to [`bat_integrate`](@ref):
```julia
integral = bat_integrate(sampleable, integration_algorithm).result
```

### Adaptive Harmonic Mean Integration (AHMI)]
BAT.jl integration algorithm type: `AHMI.AHMIntegration`, requires [AHMI.jl](https://github.com/bat/AHMI.jl).
```julia
integration_algorithm = AHMI.AHMIntegration()
```

### Vegas Integration
BAT.jl integration algorithm type: [`VEGASIntegration`](@ref) 
```julia
integration_algorithm = VEGASIntegration()
```

### Cuhre Integration
BAT.jl integration algorithm type: [`CuhreIntegration`](@ref) 
```julia
integration_algorithm = CuhreIntegration()
```

### Divonne Integration
BAT.jl integration algorithm type: [`DivonneIntegration`](@ref) 
```julia
integration_algorithm = DivonneIntegration()
```

### Integration via Bridge Sampling (experimental)
BAT.jl integration algorithm type: [`BridgeSampling`](@ref) 
```julia
integration_algorithm = BridgeSampling()
```


## Mode finding algorithms:
All following mode finding algorithms can be passed to [`bat_findmode`](@ref):
```julia
mode = bat_findmode(sampleable, modefinding_algorithm).result
```

### Nelder-Mead Optimization
BAT.jl mode finding algorithm type: [`MaxDensityNelderMead`](@ref) 
```julia
modefinding_algorithm = MaxDensityNelderMead()
```

### LBFGS Optimization
BAT.jl mode finding algorithm type: [`MaxDensityLBFGS`](@ref) 
```julia
modefinding_algorithm = MaxDensityLBFGS()
```

### Maximum Sample Estimator
BAT.jl mode finding algorithm type: [`MaxDensitySampleSearch`](@ref) 
```julia
modefinding_algorithm = MaxDensitySampleSearch()
```
