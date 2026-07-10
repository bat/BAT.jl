# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# MGVI pulls in LinearSolve, whose hard dependency
# SparseColumnPivotedQR crashes on 32-bit platforms during
# precompilation (SciML/SparseColumnPivotedQR.jl#67), so MGVI is not
# a static test dependency and its tests only run on 64-bit platforms:
if Sys.WORD_SIZE == 64
    import Pkg
    Base.identify_package("MGVI") === nothing && Pkg.add("MGVI")
    include("test_mgvi_impl.jl")
else
    @info "Skipping MGVI tests on 32-bit platform"
end
