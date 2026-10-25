# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Run from an instantiated optional project. Julia 1.10 users must first run
# Pkg.develop(path = abspath("../../..")) from this directory: [sources] is newer.
# Every potentially hanging case has a deadline enforced by its parent process.
using Test

const CASES = [
    "lifecycle_healthy", "lifecycle_callback", "lifecycle_convergence", "lifecycle_owner",
    "lifecycle_sampler", "lifecycle_producer", "lifecycle_snapshot", "lifecycle_listener",
    "lifecycle_timeout", "lifecycle_transform", "lifecycle_publication", "lifecycle_stop_publication", "lifecycle_startup",
    "lifecycle_view", "lifecycle_interactive_interrupt", "lifecycle_figure", "lifecycle_grid", "core", "buffering", "dwell", "rng", "windows",
    "static_result", "precompile_workload", "precompile_failure", "warmup",
    "domain", "domain_publication", "view_state", "api_boundaries", "numerics", "bounded_kde", "bounded_bandwidth", "kde_integration", "unbounded_kde", "unbounded_kde_integration",
    "history_release", "history_diagnostics", "release_lifecycle", "weights", "weight_failure", "hexbin", "hexbin_lifecycle", "histogram_regions", "histogram_colors", "histogram_viewport", "histogram_bars", "preinit_dispose",
    "task_rng", "hmc_rng", "renderer_failure", "scatter_admission", "failure_ownership", "display_schema", "display_live", "display_failures",
]
const ROOT = normpath(joinpath(@__DIR__, "../../.."))

if length(ARGS) == 2 && ARGS[1] == "--child"
    case = ARGS[2]
    case in CASES || error("Unknown Makie test case: $case")
    println(stderr, "Makie case $case: loading packages")
    flush(stderr)
    using BAT, CairoMakie, Makie
    isnothing(Base.get_extension(BAT, :BATMakieExt)) && error("BATMakieExt failed to load")
    realpath(dirname(dirname(pathof(BAT)))) == realpath(ROOT) ||
        error("Optional tests loaded a different BAT checkout: $(pathof(BAT))")
    CairoMakie.activate!()
    Makie.inline!(true)
    println(stderr, "Makie case $case: packages loaded; Julia $VERSION, $(Threads.nthreads(:default)) worker threads; starting assertions")
    flush(stderr)
    if case == "core"
        include(joinpath(ROOT, "test/samplers/mcmc/test_visualizer_lifecycle.jl"))
        include(joinpath(ROOT, "test/samplers/mcmc/test_mcmc_owner.jl"))
        include(joinpath(ROOT, "test/samplers/mcmc/test_mcmc_owner_parity.jl"))
    elseif case in ("history_release", "history_diagnostics")
        ENV["BAT_MAKIE_HISTORY_CASE"] = case == "history_release" ? "release" : "diagnostics"
        include(joinpath(ROOT, "test/plotting/test_makie_history_release.jl"))
    elseif startswith(case, "lifecycle_")
        ENV["BAT_MAKIE_LIFECYCLE_CASE"] = case[length("lifecycle_")+1:end]
        include(joinpath(ROOT, "test/plotting/test_makie_lifecycle.jl"))
    else
        include(joinpath(ROOT, "test/plotting/test_makie_$case.jl"))
    end
    println(stderr, "Makie case $case: finished")
    flush(stderr)
else
    selected = isempty(ARGS) ? CASES : ARGS
    all(in(CASES), selected) || error("Unknown cases; available: $(join(CASES, ", "))")
    deadline = parse(Float64, get(ENV, "BAT_MAKIE_CASE_TIMEOUT", "600"))
    isfinite(deadline) && deadline > 0 || error("BAT_MAKIE_CASE_TIMEOUT must be finite and positive")
    project = dirname(Base.active_project())
    logs = mktempdir(; cleanup = false)
    @info "Makie subprocess logs" logs
    flush(stderr)
    @testset "BAT optional Makie lifecycle and regressions" begin
        for case in selected
            @testset "$case" begin
                logfile = joinpath(logs, "$case.log")
                cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads(:default)) --project=$project $(@__FILE__) --child $case`
                process = open(logfile, "w") do io
                    run(pipeline(cmd; stdout = io, stderr = io); wait = false)
                end
                finished = timedwait(() -> process_exited(process), deadline; pollint = 0.1) == :ok
                if !finished
                    kill(process, Base.SIGKILL)
                end
                wait(process)
                passed = finished && success(process)
                if !passed
                    print(stderr, read(logfile, String))
                    @error "Makie subprocess failed" case finished exitcode = process.exitcode logfile
                end
                @test finished
                @test passed
            end
        end
    end
end
