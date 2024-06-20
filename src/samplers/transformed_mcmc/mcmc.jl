using AffineMaps

#include("mcmc_utils.jl")

#include("mcmc_weighting.jl")
include("proposaldist.jl")
#include("mcmc_sampleid.jl")
include("mcmc_algorithm.jl")
#include("mcmc_stats.jl")
include("mcmc_sample.jl")
include("tempering.jl")
include("mcmc_iterate.jl")
include("mcmc_tuning/mcmc_tuning.jl")
include("chain_pool_init.jl")
include("mcmc_convergence.jl")
include("multi_cycle_burnin.jl")
