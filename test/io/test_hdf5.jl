# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, ValueShapes

# Test only on 64-bit, automated installation of HDF5 doesn't seem to work
# properly on 32-bit on CI system:
if Int == Int64
    import HDF5

    @testset "hdf5" begin
        mktempdir() do tmp_datadir
            results_filename = joinpath(tmp_datadir, "results.hdf5")
            samples = bat_sample(BAT.example_posterior(), TransformedMCMC(nsteps = 1000, strict = false)).result
            bat_write(results_filename, samples)
            samples2 = bat_read(results_filename).result
            @test samples == samples2
        end

        @testset "nested named tuples" begin
            mktempdir() do tmp_datadir
                results_filename = joinpath(tmp_datadir, "nested.hdf5")
                prior = distprod(
                    x = (a = Uniform(0, 1), k = [4.0, 5.0], b = (c = Normal(), d = MvNormal([1.0 0.0; 0.0 1.0]))),
                    e = Uniform(0, 1)
                )
                samples = bat_sample(prior, IIDSampling(nsamples = 20)).result
                bat_write(results_filename, samples)
                samples2 = bat_read(results_filename).result
                @test varshape(samples2) == varshape(samples)
                @test samples == samples2

                # Nested shapes with ShapedAsNT value type are written as nested tables too:
                snt_filename = joinpath(tmp_datadir, "nested_snt.hdf5")
                prior_snt = NamedTupleDist(x = NamedTupleDist(ShapedAsNT, a = Uniform(0, 1), b = Normal()), e = Uniform(0, 1))
                samples_snt = bat_sample(prior_snt, IIDSampling(nsamples = 20)).result
                bat_write(snt_filename, samples_snt)
                samples_snt2 = bat_read(snt_filename).result
                @test unshaped.(samples_snt2.v) == unshaped.(samples_snt.v)
                @test samples_snt2.v.x.a == samples_snt.v.x.a
            end
        end
    end
end
