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

BAT sampling algorithm type: [`TransformedMCMC`](@ref), MCMC algorithm subtype: [`RandomWalk`](@ref)

```julia
bat_sample(target, TransformedMCMC(proposal = RandomWalk(), nsteps = 10^5, nchains = 4))
```


### Hamiltonian MC

BAT sampling algorithm type: [`TransformedMCMC`](@ref), MCMC algorithm subtype: [`HamiltonianMC`](@ref)

```julia
import ForwardDiff
set_batcontext(ad = ForwardDiff)
bat_sample(target, TransformedMCMC(proposal = HamiltonianMC()))
```


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


### Elliptical slice sampling (experimental)

BAT sampling algorithm type: [`EllipticalSliceMCMCSampling`](@ref)

```julia
import EllipticalSliceSampling
bat_sample(posterior, EllipticalSliceMCMCSampling(nsamples = 10^4))
```

BAT maps the prior to a standard Gaussian before sampling. Requires the
[EllipticalSliceSampling](https://github.com/TuringLang/EllipticalSliceSampling.jl) Julia package to be loaded explicitly.


### Slice sampling (experimental)

BAT sampling algorithm type: [`SliceMCMCSampling`](@ref)

```julia
import SliceSampling
bat_sample(target, SliceMCMCSampling(nsamples = 10^4))
```

Pass any SliceSampling.jl sampler through the `sampler` field. Requires the
[SliceSampling](https://github.com/TuringLang/SliceSampling.jl) Julia package to be loaded explicitly.


### Importance sampling (experimental)

Requires ImportanceSamplers.jl 0.0.2.
The proposal below lives in BAT's default normal-transformed coordinates:

```julia
using BAT, DensityInterface, Distributions, ImportanceSamplers, ValueShapes

prior = NamedTupleDist(rate=Gamma(2, 0.5), fraction=Beta(2, 3))
posterior = PosteriorMeasure(logfuncdensity(θ -> -θ.rate * θ.fraction), prior)
sampler = ImportanceSampling(SphericalGaussian(zeros(2), 1.5); nsamples=10_000)
evaluated = evalmeasure(posterior, ImportanceSamplersSampling(; sampler))
samples = samplesof(evaluated)
native = evalinfo(evaluated).result.samples
```

The same wrapper accepts adaptive engine algorithms such as `AMIS` or
`DeterministicMixturePMC`; they retain their own
round counts and sample budgets. Gradient methods use the AD backend selected
in `BATContext`. Its compute unit selects CPU or device execution. BAT retains weighted samples in the
original parameter shape and the estimated target mass. Native log weights,
round/proposal IDs and diagnostics remain available through `native`.

For CUDA, choose the device explicitly. This constrained example uses transports
whose scalar functions compile on CUDA:

```julia
using CUDA, MLDataDevices, HeterogeneousComputing, Random

device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
context = BATContext(cunit=DeviceUnit(device), rng=Xoshiro(42))
prior = NamedTupleDist(rate=LogNormal(0f0, 0.5f0), fraction=Uniform(0f0, 1f0))
posterior = PosteriorMeasure(logfuncdensity(θ -> -θ.rate * θ.fraction), prior)
sampler = ImportanceSampling(SphericalGaussian(zeros(Float32, 2), 1.5f0); nsamples=10_000)
evaluated = evalmeasure(posterior, ImportanceSamplersSampling(; sampler), context)
host_samples = MLDataDevices.cpu_device()(samplesof(evaluated))
```

Sampling, coordinate maps, target values and normalized weights stay on-device.
The host coordinates rounds and reads aggregate summaries. The target and its
data must support GPU execution. Use typed density objects with `Adapt` rules
for data arrays, rather than closures that capture host arrays. GPU support for
each prior transform depends on its special functions. BAT does not silently
transfer samples to CPU to evaluate an unsupported target or transform.

For vector-Gaussian priors, load `Enzyme` and set `ad=Enzyme` in the device
context. Gaussian transforms and prior scores use batched matrix operations.
The likelihood and its gradient execute on-device.

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
bat_integrate(EvaluatedMeasure(target, empirical = smpls), BridgeSampling())
```


## Mode finding algorithms

BAT function: [`bat_findmode`](@ref)


### Optim.jl Optimization Algorithms

BAT mode finding algorithm type: [`TransformedMaxDensity`](@ref) with an
[`OptimAlg`](@ref) backend.

```julia
using Optim
bat_findmode(target, TransformedMaxDensity(optalg = OptimAlg(optalg = Optim.NelderMead())))

import ForwardDiff
set_batcontext(ad = ForwardDiff)
bat_findmode(target, TransformedMaxDensity(optalg = OptimAlg(optalg = Optim.LBFGS())))
```

Bare backend configurations and raw Optim.jl optimizers are auto-wrapped in
a `TransformedMaxDensity` with default settings, so
`bat_findmode(target, Optim.NelderMead())` works as well.

Requires the [Optim](https://github.com/JuliaNLSolvers/Optim.jl) Julia package to be loaded explicitly.


### OptimizationBase.jl Optimization Algorithms

BAT mode finding algorithm type: [`TransformedMaxDensity`](@ref) with an
[`OptimizationAlg`](@ref) backend.

```julia
using OptimizationOptimJL

alg = TransformedMaxDensity(optalg = OptimizationAlg(;
    optalg = OptimizationOptimJL.ParticleSwarm(n_particles=10),
    maxiters=200,
    kwargs=(f_calls_limit=50,)
))
bat_findmode(target, alg)
```
Requires the desired package that implements the [OptimizationBase.jl](https://github.com/SciML/OptimizationBase.jl) interface to be loaded (e.g. via `import OptimizationOptimJL`).

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
