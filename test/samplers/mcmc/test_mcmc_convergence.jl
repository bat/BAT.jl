# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using ValueShapes

function _convergence_chain(values; weights = ones(Int, length(values)), chainid = 1, walkerid = 1)
    samples = [Float64.(value isa Real ? [value] : value) for value in values]
    walkerids = walkerid isa Integer ? fill(walkerid, length(samples)) : walkerid
    info = [BAT.MCMCSampleID(chainid, walkerids[i], 1, i, 1, true) for i in eachindex(samples)]
    DensitySampleVector(samples, zeros(length(samples)); weight = weights, info = info)
end

@testset "rank-normalized R-hat convergence" begin
    algorithm = RankNormalizedRhatConvergence()
    context = BATContext()

    mixed_chains = [
        _convergence_chain([1, 2, 3, 4, 1, 2, 3, 4]),
        _convergence_chain([3, 4, 1, 2, 3, 4, 1, 2], chainid = 2),
    ]
    location_mismatch = [
        _convergence_chain([1, 2, 3, 4, 1, 2, 3, 4]),
        _convergence_chain([11, 12, 13, 14, 11, 12, 13, 14], chainid = 2),
    ]
    scale_mismatch = [
        _convergence_chain([-1, 1, -1, 1, -1, 1, -1, 1]),
        _convergence_chain([-10, 10, -10, 10, -10, 10, -10, 10], chainid = 2),
    ]
    compressed = [
        _convergence_chain([1, 2, 3, 4], weights = [2, 2, 2, 2]),
        _convergence_chain([3, 4, 1, 2], weights = [2, 2, 2, 2], chainid = 2),
    ]
    expanded = [
        _convergence_chain([1, 1, 2, 2, 3, 3, 4, 4]),
        _convergence_chain([3, 3, 4, 4, 1, 1, 2, 2], chainid = 2),
    ]
    fractional = [
        _convergence_chain([1, 2, 3, 4], weights = [1.5, 2, 2, 2]),
        _convergence_chain([3, 4, 1, 2], weights = [1.0, 1.0, 1.0, 1.0], chainid = 2),
    ]
    negative = [
        _convergence_chain([1, 2, 3, 4], weights = [-1, 2, 2, 2]),
        _convergence_chain([3, 4, 1, 2], chainid = 2),
    ]
    multivariate = [
        _convergence_chain([[x, 2x, -x] for x in [1, 2, 3, 4, 1, 2, 3, 4]]),
        _convergence_chain([[x, 2x, -x] for x in [3, 4, 1, 2, 3, 4, 1, 2]], chainid = 2),
    ]
    shape = NamedTupleShape(x = ScalarShape{Real}(), y = ArrayShape{Real}(2))
    shaped = broadcast.(Ref(shape), multivariate)
    trajectory = [0, 1, 0, 1, 10, 11, 10, 11]
    walker_chains = [
        _convergence_chain(repeat(trajectory, 2), chainid = chainid,
            walkerid = repeat(1:2, inner = length(trajectory)))
        for chainid in 1:2
    ]
    plain_chains = [
        DensitySampleVector(chain.v, chain.logd; weight = chain.weight)
        for chain in mixed_chains
    ]
    empty_chain = mixed_chains[1][Int[]]
    five_draw_chains = [collect(1:5), collect(6:10)]

    @test convert(Bool, bat_convergence(mixed_chains, algorithm, context).result)
    @test !convert(Bool, bat_convergence(location_mismatch, algorithm, context).result)
    @test !convert(Bool, bat_convergence(scale_mismatch, algorithm, context).result)
    @test bat_convergence(compressed, algorithm, context).result.value ≈
          bat_convergence(expanded, algorithm, context).result.value
    @test_throws ArgumentError bat_convergence(fractional, algorithm, context)
    @test_throws ArgumentError bat_convergence(negative, algorithm, context)
    @test bat_convergence(reduce(vcat, mixed_chains), algorithm, context).result.value ≈
          bat_convergence(mixed_chains, algorithm, context).result.value
    @test bat_convergence(shaped, algorithm, context).result.value ≈
          bat_convergence(multivariate, algorithm, context).result.value
    @test bat_convergence(reduce(vcat, shaped), algorithm, context).result.value ≈
          bat_convergence(multivariate, algorithm, context).result.value
    @test !convert(Bool, bat_convergence(walker_chains, algorithm, context).result)
    @test bat_convergence(plain_chains, algorithm, context).result.value ≈
          bat_convergence(mixed_chains, algorithm, context).result.value
    @test_throws ArgumentError bat_convergence([empty_chain, mixed_chains[2]], algorithm, context)
    @test_throws ArgumentError bat_convergence([empty_chain, empty_chain], algorithm, context)
    @test collect.(BAT._split_chains(reduce(vcat, five_draw_chains), 5, 2)) ==
          [[1, 2], [6, 7], [4, 5], [9, 10]]
end
