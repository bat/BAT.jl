export AHMC

struct AHMC
    metric::HMCMetric
    gradient::Module
    integrator::HMCIntegrator
    proposal::HMCProposal
    adaptor::HMCAdaptor
end

"""
function AHMC(;
    metric = DiagEuclideanMetric(),
    gradient = ForwardDiff,
    integrator = Leapfrog(),
    proposal = NUTS(),
    adaptor = StanHMCAdaptor()
)


# Keyword arguments

Also see [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) for more detailed information on the following HMC related keyword arguments.
## Metric
options:
- `DiagEuclideanMetric()`
- `UnitEuclideanMetric()`
- `DenseEuclideanMetric()`

default: `metric = DiagEuclideanMetric()`

## Integrator
options:
- `LeapfrogIntegrator(step_size::Float64 = 0.0)`
- `JitteredLeapfrogIntegrator(step_size::Float64 = 0.0, jitter_rate::Float64 = 1.0)`
- `TemperedLEapfrogIntegrator(step_size::Float64 = 0.0, tempering_rate::Float64 = 1.05)`

default: `integrator = LeapfrogIntegrator(step_size::Float64 = 0.0)`
For `step_size = 0.0`, the initial stepsize is determined using `AdvancedHMC.find_good_eps()`


## Proposal
options:
- `FixedStepNumber(n_steps::Int64 = 10)`
- `FixedTrajectoryLength(trajectory_length::Float64 = 2.0)`
- `NUTS(sampling::Symbol, nuts::Symbol)`
with
    - `sampling =` `:SliceTS` or `:MultinomialTS`
    - `nuts = ` `:ClassicNoUTurn` or  `:GeneralisedNoUTUrn`

default: `proposal = NUTS(sampling::Symbol = :MultinomialTS, nuts::Symbol = :ClassicNoUTurn)`

## Adaptor
options:
- `MassMatrixAdaptor()`
- `StepSizeAdaptor(step_size::Float64 = 0.8)`
- `NaiveHMCAdaptor(step_size::Float64 = 0.8)`
- `StanHMCAdaptor(step_size::Float64 = 0.8)`

default: `adaptor =  StanHMCAdaptor(step_size::Float64 = 0.8)`
"""
function AHMC(;
    metric = DiagEuclideanMetric(),
    gradient = ForwardDiff,
    integrator = LeapfrogIntegrator(),
    proposal = NUTS(),
    adaptor = StanHMCAdaptor()
)

 AHMC(metric, gradient, integrator, proposal, adaptor)
end



"""
function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::AHMC;
    initial_v::Array{Array{Float64,1},1} = [rand(getprior(posterior)) for i in 1:n[2]],
    n_adapts::Int = min(div(n[1], 10), 1_000),
    verbose::Bool = true,
    drop_warmup::Bool = true
)

Sample posterior via Hamiltonian Monte Carlo using [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl).

"""
function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::AHMC;
    initial_v::Array{Array{Float64,1},1} = [rand(getprior(posterior)) for i in 1:n[2]],
    n_adapts::Int = min(div(n[1], 10), 1_000),
    verbose::Bool = true,
    drop_warmup::Bool = true
)

    dim = length(initial_v[1])
    metric = AHMCMetric(algorithm.metric, dim)


    logval_posterior(v) = density_logval(posterior, v)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logval_posterior, algorithm.gradient)

    n_samples = n[1]
    n_chains = n[2]
    sample_arr = Vector{Array{Array{Float64, 1},1}}(undef, n_chains)
    stats_arr =  Vector{Array{NamedTuple, 1}}(undef, n_chains)

    Threads.@threads for i in 1:n_chains
        algorithm.integrator.step_size == 0.0 ? algorithm.integrator.step_size = AdvancedHMC.find_good_eps(hamiltonian, initial_v[i]) : nothing
        bat_integrator = AHMCIntegrator(algorithm.integrator)

        bat_proposal = AHMCProposal(algorithm.proposal, bat_integrator)
        bat_adaptor = AHMCAdaptor(algorithm.adaptor, metric, bat_integrator)

        # sample using AdvancedHMC
        samples, stats = AdvancedHMC.sample(
            hamiltonian,
            bat_proposal,
            initial_v[i],
            n_samples,
            bat_adaptor,
            n_adapts;
            progress=false,
            verbose=verbose,
            drop_warmup = drop_warmup
        )

        sample_arr[i] = samples
        stats_arr[i] = stats
    end

    samples = vcat(sample_arr...)
    stats = vcat(stats_arr...)

    bat_samples = convert_to_bat_samples(samples, posterior)

    return (result = bat_samples, chains = stats)
end
