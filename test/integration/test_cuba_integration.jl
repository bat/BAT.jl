using BAT
using Test

import Cuba

using Base.Threads
using Distributions, ValueShapes


@testset "cuba_integration" begin
    function test_integration(target, algorithm::BAT.CubaIntegration)
        r = bat_integrate(target, algorithm).result
        val_expected = 1
        @test isapprox(r.val, val_expected, rtol = 20 * algorithm.rtol)
        # @test r.err < 100 * abs(r.val - val_expected)
    end

    uvprior = truncated(Normal(), -1, 3)

    mvprior = NamedTupleDist(
        a = Beta(0.5, 0.5),
        b = truncated(Normal(), -1, 3),
        c = Beta(5, 1),
        e = Uniform(-4, 7)
    )

    mvprior_simple = NamedTupleDist(
        a = truncated(Normal(), -1, 3),
        b = truncated(Normal(), 1, 3),
    )

    test_integration(uvprior, VEGASIntegration(trafo = NoDensityTransform(), rtol = 1e-3))
    test_integration(uvprior, VEGASIntegration(trafo = NoDensityTransform(), rtol = 1e-3, log_density_shift = 10))
    test_integration(mvprior, VEGASIntegration(trafo = NoDensityTransform(), rtol = 1e-3, nthreads = 1))
    test_integration(mvprior, VEGASIntegration(trafo = NoDensityTransform(), rtol = 1e-3, nthreads = nthreads()))
    test_integration(mvprior, VEGASIntegration(nthreads=2))
    test_integration(mvprior, VEGASIntegration())

    test_integration(mvprior, SuaveIntegration(trafo = NoDensityTransform(), rtol = 1e-3))

    test_integration(mvprior_simple, DivonneIntegration(trafo = NoDensityTransform(), rtol = 1e-3))

    test_integration(mvprior, CuhreIntegration(trafo = NoDensityTransform(), rtol = 1e-3))
    
    @test_throws ErrorException test_integration(mvprior, CuhreIntegration(maxevals=4))
    test_integration(mvprior, CuhreIntegration(maxevals=4, strict=false))
end