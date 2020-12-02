# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, PDMats
using ArraysOfArrays
using ParallelProcessingTools


if Sys.isunix()
    @testset "external_density" begin
        test_src_dir = @__DIR__
        c_src_file = joinpath(test_src_dir, "external_mvnormal_density.cxx")

        dist = MvNormal([1.0 1.5; 1.5 4.0])

        mktempdir() do tmp_build_dir
            test_density_server_binary = joinpath(tmp_build_dir, "external_mvnormal_density")
            compile_cmd = `g++ -O2 $c_src_file -o $test_density_server_binary`
            @info "Running compile command $compile_cmd"
            run(compile_cmd)
            @info isfile(test_density_server_binary)
            likelihood = BAT.ExternalDensity(`$test_density_server_binary`, 0)

            @test begin
                x = [1.23, 2.34]
                llvalue = BAT.eval_logval_unchecked(likelihood, x)
                llvalue ≈ logpdf(dist, x)
            end

            @test begin
                n = 100
                params = nestedview(rand(2, n) .* 4 .- 2)
                ll_external = zeros(n)
                @sync for i in eachindex(params, ll_external)
                    @mt_async ll_external[i] = BAT.eval_logval_unchecked(likelihood, params[i])
                end
                ll_expected = logpdf.((dist,), params)
                ll_external ≈ ll_expected
            end
  
            close(likelihood)
        end
    end
end
