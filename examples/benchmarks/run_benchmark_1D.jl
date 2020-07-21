using BAT, ValueShapes, IntervalSets, Distributions, Plots
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables, HypothesisTests, Statistics

include("utils.jl")
include("functions_1D.jl")

algorithm = MetropolisHastings()
n_chains = 8
n_samples = 10^5

for i in 1:length(testfunctions_1D)
    sample_stats_all = run1D(
        collect(keys(testfunctions_1D))[i], #There might be a nicer way but I need the name to save the plots
        testfunctions_1D,
        sample_stats[i],
        run_stats[i]
    )
end
make_1D_results(testfunctions_1D,sample_stats)
save_stats_1D(collect(keys(testfunctions_1D)),run_stats,run_stats_names)
