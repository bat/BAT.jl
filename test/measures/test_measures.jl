# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "measures" begin
    include("test_bat_dist_measure.jl")
    include("test_density_sample_measure.jl")
    include("test_bat_pwr_measure.jl")
    include("test_bat_pushfwd_measure.jl")
    include("test_bat_weighted_measure.jl")
    include("test_truncate_batmeasure.jl")
    include("test_measure_functions.jl")
end
