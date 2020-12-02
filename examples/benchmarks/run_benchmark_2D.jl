function run_2D_benchmark(;algorithm = MetropolisHastings(),n_chains = 8,n_steps = 10^5)
    for i in 1:length(testfunctions_2D)
        sample_stats_all = run2D(
            collect(keys(testfunctions_2D))[i], #There might be a nicer way but I need the name to save the plots
            testfunctions_2D,
            sample_stats2D[i],
            run_stats2D[i],
            algorithm,
            n_steps,
            n_chains
        )
    end
    make_2D_results(testfunctions_2D,sample_stats2D)
    save_stats_2D(collect(keys(testfunctions_2D)),run_stats2D,run_stats_names2D)
end
