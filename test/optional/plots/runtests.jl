# This file is a part of BAT.jl, licensed under the MIT License (MIT).

ENV["GKSwstype"] = "100"
using BAT

const ROOT = normpath(joinpath(@__DIR__, "../../.."))
realpath(dirname(dirname(pathof(BAT)))) == realpath(ROOT) ||
    error("Optional tests loaded a different BAT checkout: $(pathof(BAT))")
include(joinpath(ROOT, "test/plotting/test_plots_weight_hardening.jl"))
include(joinpath(ROOT, "test/plotting/test_plots_histogram_regions.jl"))
