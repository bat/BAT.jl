# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "samplers" begin
    include("test_bat_sample.jl")
    include("test_evaluated_measure.jl")
    include("test_pathfinder.jl")
    include("mcmc/test_mcmc.jl")
    include("importance/test_importance_sampler.jl")
    # MGVI pulls in LinearSolve, whose hard dependency
    # SparseColumnPivotedQR crashes on 32-bit platforms during
    # precompilation (SciML/SparseColumnPivotedQR.jl#67), so MGVI is not
    # a static test dependency and its tests only run on 64-bit platforms:
    if Sys.WORD_SIZE == 64
        import Pkg
        Base.identify_package("MGVI") === nothing && Pkg.add("MGVI")
        include("test_mgvi.jl")
    else
        @info "Skipping MGVI tests on 32-bit platform"
    end
end
