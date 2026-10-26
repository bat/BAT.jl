# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, ValueShapes
using ArraysOfArrays, FillArrays, StructArrays
import MeasureBase
using MeasureBase: productmeasure, asmeasure, StdUniform, StdNormal

# Test only on 64-bit, automated installation of HDF5 doesn't seem to work
# properly on 32-bit on CI system:
if Int == Int64
    import HDF5

    @testset "hdf5" begin
        mktempdir() do tmp_datadir
            function roundtrip(content, name)
                filename = joinpath(tmp_datadir, name)
                bat_write(filename, content)
                return bat_read(filename).result
            end

            @testset "MCMC samples" begin
                samples = bat_sample(BAT.example_posterior(), TransformedMCMC(nsteps = 1000, strict = false)).result
                samples2 = roundtrip(samples, "results.hdf5")
                @test samples == samples2
                @test samples2.info isa BAT.MCMCSampleIDVector
            end

            @testset "structured variates" begin
                μ = productmeasure((
                    a = asmeasure(Normal()),
                    b = asmeasure(MvNormal([2.0 0.5; 0.5 1.0])),
                    c = MeasureBase.Dirac(4.2),
                    k = MeasureBase.Dirac([1, 2]),
                    d = productmeasure((x = StdUniform(), y = StdNormal()^(2, 3))),
                ))
                n = 7
                samples = DensitySampleVector(rand(μ^n), μ, logd = randn(n))
                filename = joinpath(tmp_datadir, "structured.h5")
                bat_write(filename, samples)
                samples2 = bat_read(filename).result
                @test samples2 == samples
                @test varshape(samples2) == varshape(μ)
                @test samples2.v.c isa Fill && samples2.v.k isa Fill{Vector{Int}}
                @test samples2.v.d.y isa VectorOfSimilarArrays{Float64,2}

                # Objects carry their layout and can be read on their own:
                @test bat_read(filename, "v/d").result == samples.v.d
                @test bat_read(filename, "v/c").result == Fill(4.2, n)
                HDF5.h5open(filename) do f
                    @test HDF5.read_attribute(f, "bat_layout") == "samples"
                    @test HDF5.read_attribute(f, "bat_format") == 6
                    @test HDF5.read_attribute(f["v"], "bat_fields") == ["a", "b", "c", "k", "d"]
                    @test HDF5.read_attribute(f["v/d/y"], "bat_layout") == "array"
                    @test size(f["v/d/y"]) == (2, 3, n)
                    @test HDF5.read_attribute(f["v/k"], "bat_layout") == "const"
                    @test HDF5.read_attribute(f["v/k"], "bat_length") == n
                    @test read(f["v/k"]) == [1, 2]
                end

                # Constants come from the layout, not from the values:
                samples2_1 = roundtrip(samples[1:1], "single.h5")
                @test samples2_1 == samples[1:1]
                @test varshape(samples2_1) == varshape(μ)
            end

            @testset "scalar and flat variates, groups" begin
                s_scalar = DensitySampleVector(v = randn(5), logd = randn(Float32, 5))
                s_flat = DensitySampleVector(v = nestedview(randn(3, 5)), aux = StructArray((q = randn(5),)))
                content2 = roundtrip((scalar = s_scalar, flat = s_flat), "group.h5")
                @test content2 isa NamedTuple{(:scalar, :flat)}
                @test content2.scalar == s_scalar && eltype(content2.scalar.logd) == Float32
                # Default log-densities are NaN:
                @test isequal(content2.flat, s_flat) && varshape(content2.flat) == ArrayShape{Real}(3)

                filename = joinpath(tmp_datadir, "pair.h5")
                bat_write(filename, "results/samples" => s_scalar)
                @test bat_read(filename).result.results.samples == s_scalar
                @test bat_read(filename, "results/samples").result == s_scalar
            end

            @testset "BAT 5.x files" begin
                # Written without layout attributes, constants stored densely:
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
                    info = HDF5.create_group(f, "info"; track_order = true)
                    info["chainid"] = fill(Int32(1), n)
                    info["walkerid"] = fill(Int32(2), n)
                    info["chaincycle"] = fill(Int32(1), n)
                    info["stepno"] = collect(1:n)
                    info["proposalid"] = fill(Int32(1), n)
                    info["sampletype"] = fill(true, n)
                end
                samples = bat_read(filename).result
                @test varshape(samples) == NamedTupleShape(a = ScalarShape{Real}(), m = ArrayShape{Real}(2, 3), c = ConstValueShape(4.2))
                @test samples.v.m == nestedview(M, 2)
                @test samples.v.c isa Fill
                @test samples.info isa BAT.MCMCSampleIDVector && all(isequal(2), samples.info.walkerid)
            end

            @testset "nested named tuples" begin
                prior = distprod(
                    x = (a = Uniform(0, 1), k = [4.0, 5.0], b = (c = Normal(), d = MvNormal([1.0 0.0; 0.0 1.0]))),
                    e = Uniform(0, 1)
                )
                samples = bat_sample(prior, IIDSampling(nsamples = 20)).result
                samples2 = roundtrip(samples, "nested.hdf5")
                @test varshape(samples2) == varshape(samples)
                @test samples == samples2

                # Nested shapes with ShapedAsNT value type are written as nested tables too:
                prior_snt = NamedTupleDist(x = NamedTupleDist(ShapedAsNT, a = Uniform(0, 1), b = Normal()), e = Uniform(0, 1))
                samples_snt = bat_sample(prior_snt, IIDSampling(nsamples = 20)).result
                samples_snt2 = roundtrip(samples_snt, "nested_snt.hdf5")
                @test unshaped.(samples_snt2).v == unshaped.(samples_snt).v
                @test samples_snt2.v.x.a == samples_snt.v.x.a
            end
        end
    end
end
