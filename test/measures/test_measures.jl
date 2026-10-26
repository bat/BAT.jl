# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "measures" begin
    include("test_density_sample_measure.jl")
    include("test_pushfwd_measure.jl")
    include("test_superpos_measure.jl")
    include("test_posterior_measure.jl")
    include("test_truncate_batmeasure.jl")
    include("test_evaluated_measure.jl")
    include("test_bispaced_measure.jl")
    include("test_measure_functions.jl")
end
