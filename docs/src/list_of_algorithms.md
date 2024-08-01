# List of BAT algorithms 

BAT offers multiple algorithms for sampling, integration and optimization:


## Sampling algorithms

BAT function: [`bat_sample`](@ref)


### IIDSampling

BAT sampling algorithm type: [`IIDSampling`](@ref)

```julia
bat_sample(target.prior, IIDSampling(nsamples=10^5))
```


### Metropolis-Hastings

BAT sampling algorithm type: [`MCMCSampling`](@ref), MCMC algorithm subtype: [`MetropolisHastings`](@ref)

```julia
bat_sample(target, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^5, nchains = 4))
```


### Hamiltonian MC

BAT sampling algorithm type: [`MCMCSampling`](@ref), MCMC algorithm subtype: [`HamiltonianMC`](@ref)

```julia
import AdvancedHMC, ForwardDiff
set_batcontext(ad = ADSelector(ForwardDiff))
bat_sample(target, MCMCSampling(mcalg = HamiltonianMC()))
```
Requires the [AdvancedHMC](https://github.com/TuringLang/AdvancedHMC.jl) Julia package to be loaded explicitly.


### Reactive Nested Sampling (experimental)

BAT sampling algorithm type: `ReactiveNestedSampling`

```julia
import UltraNest
bat_sample(target, ReactiveNestedSampling())
```

Requires the [UltraNest](https://github.com/bat/UltraNest.jl) Julia package to be loaded explicitly.


### Ellipsoidal Nested Sampling (experimental)

BAT sampling algorithm type: [`EllipsoidalNestedSampling`](@ref)
```julia
import NestedSamplers
bat_sample(target, EllipsoidalNestedSampling())
```

Requires the [NestedSamplers](https://github.com/TuringLang/NestedSamplers.jl) Julia package to be loaded explicitly.


### Sobol Sampler
BAT sampling algorithm type: [`SobolSampler`](@ref)

```julia
bat_sample(target, SobolSampler(nsamples=10^5))
```


### Grid Sampler

BAT sampling algorithm type: [`GridSampler`](@ref)

```julia
bat_sample(target, GridSampler(ppa=100))
```


## Prior Importance Sampler

BAT sampling algorithm type: [`PriorImportanceSampler`](@ref)

```julia
bat_sample(target, PriorImportanceSampler(nsamples=10^5))
```


## Integration algorithms

BAT function: [`bat_integrate`](@ref)

### Vegas Integration

BAT integration algorithm type: [`VEGASIntegration`](@ref)

```julia
import Cuba
bat_integrate(target, VEGASIntegration())
```

Requires the [Cuba](https://github.com/giordano/Cuba.jl) Julia package to be loaded explicitly.


### Suave Integration

BAT integration algorithm type: [`SuaveIntegration`](@ref)

```julia
import Cuba
bat_integrate(target, SuaveIntegration())
```

Requires the [Cuba](https://github.com/giordano/Cuba.jl) Julia package to be loaded explicitly.


### Cuhre Integration

BAT integration algorithm type: [`CuhreIntegration`](@ref)

```julia
import Cuba
bat_integrate(target, CuhreIntegration())
```
Requires the [Cuba](https://github.com/giordano/Cuba.jl) Julia package to be loaded explicitly.


### Divonne Integration

BAT integration algorithm type: [`DivonneIntegration`](@ref) 

```julia
import Cuba
bat_integrate(target, DivonneIntegration())
```
Requires the [Cuba](https://github.com/giordano/Cuba.jl) Julia package to be loaded explicitly.


### Integration via Bridge Sampling (experimental)

BAT integration algorithm type: [`BridgeSampling`](@ref) 

```julia
bat_integrate(EvaluatedMeasure(target, smpls), BridgeSampling())
```


## Mode finding algorithms

BAT function: [`bat_findmode`](@ref)


### Optim.jl Optimization Algorithms

BAT mode finding algorithm type: [`OptimAlg`](@ref).

```julia
using Optim
bat_findmode(target, OptimAlg(optalg = Optim.NelderMead()))

import ForwardDiff
set_batcontext(ad = ADSelector(ForwardDiff))
bat_findmode(target, OptimAlg(optalg = Optim.LBFGS()))
```

Requires the [Optim](https://github.com/JuliaNLSolvers/Optim.jl) Julia package to be loaded explicitly.


### Optimization.jl Optimization Algorithms

BAT mode finding algorithm type: [`OptimizationAlg`](@ref).

```julia
using OptimizationOptimJL

alg = OptimizationAlg(; 
    optalg = OptimizationOptimJL.ParticleSwarm(n_particles=10), 
    maxiters=200, 
    kwargs=(f_calls_limit=50,)
)
bat_findmode(target, alg)
```
Requires one of the [Optimization.jl](https://github.com/SciML/Optimization.jl) packages to be loaded explicitly.

### Maximum Sample Estimator

BAT mode finding algorithm type: [`MaxDensitySearch`](@ref) 

```julia
bat_findmode(smpls, MaxDensitySearch())
```


## File-I/O

### Plain HDF5

BAT I/O algorithm type: [`BATHDF5IO`](@ref) 

```julia
import HDF5
bat_write("results.h5", smpls)

# ... later ...

smpls = bat_read("results.h5").result
```

### JLD2

Not BAT-specific, JLD2 is able to handle complex Julia data structures in
general.

```julia
using FileIO
import JLD2
FileIO.save("results.jld2", Dict("smpls" => smpls))

# ... later ...

smpls = FileIO.load("results.jld2", "smpls")
```
