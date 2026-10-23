# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, ValueShapes, ArraysOfArrays

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
                @test unshaped.(samples_snt2).v == unshaped.(samples_snt).v
                @test samples_snt2.v.x.a == samples_snt.v.x.a
            end
        end

        @testset "BAT 5.x files" begin
            mktempdir() do tmp_datadir
                filename = joinpath(tmp_datadir, "bat5.h5")
                n = 5
                M = reshape(collect(1.0:6n), 2, 3, n)
                HDF5.h5open(filename, "w"; track_order = true) do f
                    v = HDF5.create_group(f, "v"; track_order = true)
                    v["a"] = collect(1.0:n)
                    v["m"] = M
                    v["c"] = fill(4.2, n)
                    f["logd"] = randn(n)
                    f["weight"] = fill(1, n)
                end
                samples = bat_read(filename).result
                @test varshape(samples) == NamedTupleShape(a = ScalarShape{Real}(), m = ArrayShape{Real}(2, 3), c = ConstValueShape(4.2))
                @test samples.v.m == nestedview(M, 2)
            end
        end
    end
end
