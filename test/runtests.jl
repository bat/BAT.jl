# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

@info "Running tests with $(Base.Threads.nthreads()) Julia threads active."

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger(stderr, Logging.Error))

import AbstractMCMC
AbstractMCMC.setprogress!(false)

# Tests are divided into groups of roughly equal runtime that CI can run in
# parallel. The environment variable BAT_TEST_GROUP selects the groups to
# run, as a comma-separated list of group names or "all" (the default).
const test_groups = [
    "core" => [
        "test_aqua.jl",
        "utils/test_utils.jl",
        "rngs/test_rngs.jl",
        "distributions/test_distributions.jl",
        "variates/test_variates.jl",
        "transforms/test_transforms.jl",
        "densities/test_densities.jl",
    ],
    "stats" => [
        "measures/test_measures.jl",
        "initvals/test_initvals.jl",
        "statistics/test_statistics.jl",
        "optimization/test_optimization.jl",
        "io/test_io.jl",
        "plotting/test_plotting.jl",
        "integration/test_integration.jl",
    ],
    "mcmc" => [
        "samplers/test_bat_sample.jl",
        "samplers/test_pathfinder.jl",
        "samplers/mcmc/test_proposaldist.jl",
        "samplers/mcmc/test_mcmc_defaults.jl",
        "samplers/mcmc/test_mcmc_sample.jl",
        "samplers/mcmc/test_mh.jl",
        "samplers/mcmc/test_ram_tuner.jl",
        "samplers/mcmc/test_fisher_tuner.jl",
        "samplers/importance/test_importance_sampler.jl",
        "samplers/test_mgvi.jl",
    ],
    "hmc" => [
        "samplers/mcmc/test_hmc_nuts.jl",
        "samplers/mcmc/test_hmc.jl",
    ],
]

selected_groups = split(get(ENV, "BAT_TEST_GROUP", "all"), ",")
if !("all" in selected_groups)
    unknown_groups = setdiff(selected_groups, first.(test_groups))
    isempty(unknown_groups) || error("Unknown test group(s): ", join(unknown_groups, ", "))
end

Test.@testset "Package BAT" begin
    for (group, testfiles) in test_groups
        if "all" in selected_groups || group in selected_groups
            Test.@testset "$group" begin
                for testfile in testfiles
                    include(testfile)
                end
            end
        end
    end
end
