# This file is a part of BAT.jl, licensed under the MIT License (MIT).
# Run separately with a native display and external deadline.
using BAT, GLMakie, Makie, Test, Logging
@test realpath(pkgdir(BAT)) == realpath(joinpath(@__DIR__, "../../.."))
GLMakie.activate!(visible=false)
Makie.inline!(false)
@test_logs min_level=Logging.Warn begin
    include("../../plotting/test_makie_range_events.jl")
end
