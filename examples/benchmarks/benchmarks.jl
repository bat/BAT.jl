using BAT, ValueShapes, IntervalSets, Distributions, Plots, EmpiricalDistributions
using BATTestCases
using AHMI
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables, HypothesisTests, Statistics
bat_logdebug()

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
    #include("run_benchmark_1D.jl")
    include("run_benchmark_2D.jl")
    include("run_benchmark_ND.jl")
    end

function do_benchmarks(;algorithm=RandomWalk(), nsteps=10^6, nchains=8)
    #run_1D_benchmark(algorithm=algorithm, nsteps=nsteps, nchains=nchains)
    run_2D_benchmark(algorithm=algorithm, nsteps=nsteps, nchains=nchains)
    run_ND_benchmark(n_dim=2:2:20,algorithm=RandomWalk(), nsteps=nsteps, nchains=4)
    run_ks_ahmc_vs_mh(n_dim=20:5:35)
end

setup_benchmark()
do_benchmarks()
