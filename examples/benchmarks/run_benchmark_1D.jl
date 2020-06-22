using BAT, ValueShapes, IntervalSets, Distributions, Plots
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables

include("testDensities.jl")
include("utils.jl")
include("functions_1D.jl")

algorithm = MetropolisHastings()
n_chains = 8
n_samples = 10^5

for i in 1:length(name)
    sample_stats_all = run1D(
        posterior[i],
        name[i],
        analytical_integral[i],
        func[i],
        analytical_stats[i],
        sample_stats[i],
        run_stats[i]
    )
end
make_1D_results(name,sample_stats,analytical_stats)
save_stats_1D(name,run_stats,run_stats_names)
