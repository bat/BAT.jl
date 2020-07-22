using BAT, ValueShapes, IntervalSets, Distributions, Plots, EmpiricalDistributions
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables, HypothesisTests, Statistics

function setup_benchmark()
        if(!(("plots1D" in readdir()) && ("plots2D" in readdir()) && ("results" in readdir())))
            mkpath("benchmark_results")
            cd("benchmark_results")
            mkpath("plots1D")
            mkpath("plots2D")
            mkpath("results")
        end

        include("utils.jl")
        include("functions_1D.jl")
        include("functions_2D.jl")
        include("run_benchmark_1D.jl")
        include("run_benchmark_2D.jl")
        include("run_benchmark_ND.jl")
    end

function do_benchmarks(;algorithm=MetropolisHastings(), n_samples=10^5, n_chains=8)
    run_1D_benchmark(algorithm=algorithm, n_samples=n_samples, n_chains=n_chains)
    #run_2D_benchmark(algorithm=algorithm, n_samples=n_samples, n_chains=n_chains)
    #run_ND_benchmark(algorithm=MetropolisHastings(), n_samples=10^4, n_chains=10)
end
setup_benchmark()
do_benchmarks()
