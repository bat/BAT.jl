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
            h5_track_order_available =
                :track_order in HDF5.class_propertynames(HDF5.FileCreateProperties)
            chainid, walkerid, chaincycle, stepno, sampletype =
                Int32[2], Int32[4], Int32[6], Int64[8], Bool[true]
            # Exercise distinct migration contracts.
            for (schema, expected_walkerid, requires_tracked_order) in (
                ((;chainid, walkerid, chaincycle, stepno, sampletype), walkerid, true),
                ((;chainid, chaincycle, stepno, sampletype), Int32[1], true),
                ((;chaincycle, chainid, sampletype, stepno), Int32[1], false),
            )
                (!requires_tracked_order || h5_track_order_available) || continue
                h5_options = requires_tracked_order ? (track_order = true,) : (;)
                mktempdir() do tmp_datadir
                    filename = joinpath(tmp_datadir, "legacy-mcmc-id.h5")
                    HDF5.h5open(filename, "w"; h5_options...) do file
                        group = HDF5.create_group(file, "info"; h5_options...)
                        # Preserve legacy field order.
                        for field in propertynames(schema)
                            group[string(field)] = getproperty(schema, field)
                        end
                    end
                    ids = bat_read(filename, "info", BATHDF5IO()).result
                    expected = BAT.MCMCSampleIDVector((
                        chainid, expected_walkerid, chaincycle, stepno, Int32[1], sampletype,
                    ))
                    @test ids == expected
                end
            end
        end
    end
end
