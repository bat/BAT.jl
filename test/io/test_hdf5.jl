# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

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

        @testset "legacy MCMC sample IDs" begin
            chainid = Int32[2, 3]
            walkerid = Int32[4, 5]
            chaincycle = Int32[6, 7]
            stepno = Int64[8, 9]
            sampletype = Bool[true, false]

            legacy_schemas = (
                (
                    "without proposal ID",
                    (;chainid, walkerid, chaincycle, stepno, sampletype),
                    walkerid,
                ),
                (
                    "without walker and proposal IDs",
                    (;chainid, chaincycle, stepno, sampletype),
                    fill(Int32(1), 2),
                ),
                (
                    "without tracked field order",
                    (;chaincycle, chainid, sampletype, stepno),
                    fill(Int32(1), 2),
                ),
            )

            for (name, schema, expected_walkerid) in legacy_schemas
                @testset "$name" begin
                    mktempdir() do tmp_datadir
                        filename = joinpath(tmp_datadir, "legacy-mcmc-id.h5")
                        HDF5.h5open(filename, "w"; track_order = true) do file
                            group = HDF5.create_group(file, "info"; track_order = true)
                            for field in propertynames(schema)
                                group[string(field)] = getproperty(schema, field)
                            end
                        end

                        ids = bat_read(filename, "info", BATHDF5IO()).result
                        @test ids isa BAT.MCMCSampleIDVector
                        @test ids.chainid == chainid
                        @test ids.walkerid == expected_walkerid
                        @test ids.chaincycle == chaincycle
                        @test ids.stepno == stepno
                        @test ids.proposalid == fill(Int32(1), 2)
                        @test ids.sampletype == sampletype
                        @test map(eltype, (
                            ids.chainid,
                            ids.walkerid,
                            ids.chaincycle,
                            ids.stepno,
                            ids.proposalid,
                            ids.sampletype,
                        )) ==
                            (Int32, Int32, Int32, Int64, Int32, Bool)
                    end
                end
            end
        end
    end
end
