export AHMC

struct AHMC <: AbstractSamplingAlgorithm end


function bat_sample(
    posterior::AnyPosterior,
    n::AnyNSamples,
    algorithm::AHMC;
    n_adapts::Int = min(div(n[1], 10), 1_000),
    metric = DiagEuclideanMetric(),
    gradient = ForwardDiff,
    integrator = Leapfrog(),
    proposal = NUTS(),
    adaptor =  StanHMCAdaptor(),
    verbose::Bool = true,
    drop_warmup::Bool = true
)

    initial_v = rand(getprior(posterior))

    dim = length(initial_v)
    metric = get_AHMCmetric(metric, dim)

    logval_posterior(v) = density_logval(posterior, v)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, logval_posterior, gradient)

    if integrator.ϵ == 0
        integrator.ϵ = AdvancedHMC.find_good_eps(hamiltonian, initial_v)
    end
    integrator = get_AHMCintegrator(integrator)

    proposal = get_AHMCproposal(proposal, integrator)
    adaptor = get_AHMCAdaptor(adaptor, metric, integrator)

    n_samples = n[1]

    # call AdvancedHMC for sampling
    samples, stats = AdvancedHMC.sample(
        hamiltonian,
        proposal,
        initial_v,
        n_samples,
        adaptor,
        n_adapts;
        progress=false,
        verbose=verbose,
        drop_warmup = drop_warmup
    )

    bat_samples = convert_to_bat_samples(samples, posterior)

    return (result = bat_samples, chains = stats)
end
