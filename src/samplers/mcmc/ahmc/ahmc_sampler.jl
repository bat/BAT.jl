export AHMC

struct AHMC
    metric::AHMCMetric
    gradient::Module
    integrator::AHMCIntegrator
    proposal::AHMCProposal
    adaptor::AHMCAdaptor
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
- `Leapfrog(ϵ::Real = 0)`, with stepsize `ϵ. `
- `JitteredLeapfrog(ϵ::Real = 0, n::Real = 1.0)` with the jitter rate `n`,
- `TemperedLEapfrog(ϵ::Real = 0, a::Real = 1.05)` with tempering rate `a`

default: `integrator = Leapfrog(ϵ::Real = 0)`, with stepsize `ϵ. `
For `ϵ = 0`, the initial stepsize is determined using `AdvancedHMC.find_good_eps()`


## Proposal
options:
- `StaticTrajectory(n::Real = 10)`
- `HMCDA(len_traj::Real = 2)`
- `NUTS(sampling::Symbol, nuts::Symbol)`
with
    - `sampling =` `:SliceTS` or `:MultinomialTS`
    - `nuts = ` `:ClassicNoUTurn` or  `:GeneralisedNoUTUrn`

default: `proposal = NUTS(sampling::Symbol = :MultinomialTS, nuts::Symbol = :ClassicNoUTurn)`

## Adaptor
options:
- `Preconditioner()`
- `NesterovDualAveraging(δ::Real = 0.8)`
- `NaiveHMCAdaptor(δ::Real = 0.8)`
- `StanHMCAdaptor(δ::Real = 0.8)`

default: `adaptor =  StanHMCAdaptor(δ::Real = 0.8)`
"""
function AHMC(;
    metric = DiagEuclideanMetric(),
    gradient = ForwardDiff,
    integrator = Leapfrog(),
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
    metric = get_AHMCmetric(algorithm.metric, dim)
    n_samples = n[1]; n_chains = n[2]

    sample_arr = Vector{Array{Array{Float64, 1},1}}(undef, n_chains)
    stats_arr =  Vector{Array{NamedTuple, 1}}(undef, n_chains)

    logval_posterior(v) = density_logval(posterior, v)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logval_posterior, algorithm.gradient)


    Threads.@threads for i in 1:n_chains
        algorithm.integrator.ϵ == 0 ? algorithm.integrator.ϵ = AdvancedHMC.find_good_eps(hamiltonian, initial_v[i]) : nothing
        bat_integrator = get_AHMCintegrator(algorithm.integrator)

        bat_proposal = get_AHMCproposal(algorithm.proposal, bat_integrator)
        bat_adaptor = get_AHMCAdaptor(algorithm.adaptor, metric, bat_integrator)

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
