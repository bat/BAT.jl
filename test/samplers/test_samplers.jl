# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "samplers" begin
    include("test_bat_sample.jl")
    include("mcmc/test_mcmc.jl")
    include("importance/test_importance_sampler.jl")
    include("nested_sampling/test_nested_sampling.jl")
    include("test_mgvi.jl")
end
