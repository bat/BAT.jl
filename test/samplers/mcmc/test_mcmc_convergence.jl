# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, Distributions, Random123, Test, ValueShapes

@testset "rank-normalized R-hat" begin
    convergence = RankNormalizedRhatConvergence()
    chains(values; weight = ones(Int, length(first(values)))) = [
        DensitySampleVector([[x] for x in v], zeros(length(v)); weight) for v in values
    ]
    rhat(samples) = bat_convergence(samples, convergence).result

    mixed = chains([[1, 2, 3, 1, 2, 3], [3, 1, 2, 3, 1, 2]])
    separated = chains([[1, 2, 3, 1, 2, 3], [11, 12, 13, 11, 12, 13]])
    @test Bool(rhat(mixed))
    @test !Bool(rhat(separated))

    weighted = chains([[1, 2, 3], [3, 1, 2]]; weight = [1, 2, 3])
    repeated = chains([[1, 2, 2, 3, 3, 3], [3, 1, 1, 2, 2, 2]])
    @test rhat(weighted).value ≈ rhat(repeated).value

    odd = chains([[0, 6, 6, 6, 1], [3, 6, 6, 3, 1]])
    @test rhat(odd).value ≈ 1.0375770154825823

    @test !Bool(rhat(chains([[-Inf, 0, 1, -Inf, 0, 1] for _ in 1:2])))
    @test !Bool(rhat(chains([fill(1.0, 6) for _ in 1:2])))
    @test !Bool(rhat(chains([[1, 2, 3], [3, 1, 2]])))
    with_zero = chains([[NaN, 1, 2, 3], [NaN, 3, 1, 2]]; weight = [0, 1, 2, 3])
    @test rhat(with_zero).value ≈ rhat(weighted).value

    extreme = [typemin(Int), -1, 0, 0, 0, 1, typemax(Int), 0]
    @test rhat(chains([extreme, reverse(extreme)])).value ≈
        rhat(chains([Float64.(extreme), Float64.(reverse(extreme))])).value
    scaled = [[-1, -0.5, 0, 0.5, 1, 0.8, 0.9, 0.7], [1, 0.5, 0, -0.5, -1, 0.7, 0.8, 0.9]]
    @test rhat(chains([floatmax(Float64) .* v for v in scaled])).value ≈ rhat(chains(scaled)).value

    target = NamedTupleDist(a = Normal(), b = Normal(1, 2))
    algorithm = TransformedMCMC(;
        nchains = 2, nwalkers = 2, nsteps = 32, init = MCMCRetryInit(nsteps_init = 3),
        convergence, strict = false,
        burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 3, max_ncycles = 1, nsteps_final = 0),
    )
    samples = samplesof(evalmeasure(target, algorithm, BATContext(rng = Philox4x((530, 530)))))
    @test sum(samples.weight) == 128
    @test rhat(samples).value ≈ rhat(unshaped.(samples)).value

    path = [-4, -3, -2, -1, 1, 2, 3, 4]
    function walker(chainid, walkerid, values)
        info = [BAT.MCMCSampleID(chainid, walkerid, 1, i, 1, true) for i in eachindex(values)]
        DensitySampleVector([[x] for x in values], zeros(length(values)); info)
    end
    first_walker = [walker(c, 1, path) for c in 1:2]
    second_walker = [walker(c, 2, [1, 2, 3, 4, 4, 3, 2, 1]) for c in 1:2]
    ensembles = vcat.(first_walker, second_walker)
    expected = max(rhat(first_walker).value, rhat(second_walker).value)
    @test rhat(ensembles).value ≈ expected
    merged = vcat(ensembles...)
    permutation = [5, 3, 7, 1, 4, 6, 2, 8, 13, 11, 15, 9, 12, 14, 10, 16]
    @test rhat(merged[vcat(permutation, permutation .+ 16)]).value ≈ expected
end
