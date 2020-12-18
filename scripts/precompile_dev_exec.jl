let
    import LinearAlgebra, PDMats, PositiveFactorizations

    Sigma = PDMats.PDMat(LinearAlgebra.cholesky(PositiveFactorizations.Positive, LinearAlgebra.Hermitian([1.0  0.5; 0.5  2.0])))

    import Distributions

    dist = Distributions.MvNormal(Sigma)


    import Optim, DistributionsAD, ForwardDiff, Zygote

    f = let dist = dist
        v -> Distributions.logpdf(dist, v)
    end

    g_fw! = let f = f
        (r, v) -> r .= ForwardDiff.gradient(f, v)
    end

    g_zg! = let f = f
        (r, v) -> r .= first(Zygote.gradient(f, v))
    end

    init_x = rand(dist)

    Optim.maximize(f, init_x, Optim.NelderMead())
    Optim.maximize(f, g_fw!, init_x, Optim.LBFGS())
    Optim.maximize(f, g_zg!, init_x, Optim.LBFGS())


    import AdvancedHMC

    let
        metric = AdvancedHMC.DiagEuclideanMetric(length(dist))
        hamiltonian = AdvancedHMC.Hamiltonian(metric, f, ForwardDiff)
        initial_ϵ = AdvancedHMC.find_good_stepsize(hamiltonian, init_x)
        integrator = AdvancedHMC.Leapfrog(initial_ϵ)
        proposal = AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS, AdvancedHMC.GeneralisedNoUTurn}(integrator)
        adaptor = AdvancedHMC.StanHMCAdaptor(AdvancedHMC.MassMatrixAdaptor(metric), AdvancedHMC.StepSizeAdaptor(0.8, integrator))
        samples, stats = AdvancedHMC.sample(hamiltonian, proposal, init_x, 2000, adaptor, 1000; progress=true)
    end


    ENV["GKSwstype"] = "nul"
    import Plots

    Plots.plot(rand(100))
    Plots.histogram(randn(10^5))
    Plots.histogram2d(randn(10^5), randn(10^5))
end
