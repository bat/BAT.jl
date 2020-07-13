using BAT, ValueShapes, IntervalSets, Distributions, Plots
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables

include("utils.jl")
include("functions_2D.jl")

algorithm = MetropolisHastings()
n_chains = 10
n_samples = 10^5

for i in 1:length(name2D)
    sample_stats_all = run2D(
        posterior2D[i],
        name2D[i],
        analytical_stats2D[i],
        sample_stats2D[i],
        run_stats2D[i]
  )
end
make_2D_results(sample_stats2D,analytical_stats2D)
save_stats_2D(name2D,run_stats2D,run_stats_names2D)
