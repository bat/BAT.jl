using BAT, ValueShapes, IntervalSets, Distributions, Plots
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables

n_dim = 2:2:10

algorithm = MetropolisHastings()
n_chains = 10
n_samples = 10^4

results = Matrix{Any}(undef,length(n_dim),3+1)

for i in 1:length(n_dim)
    i_dim = n_dim[i]
    results[i,1] = i_dim
    sig = Matrix{Float64}([1.5^2 1.5*2.5*0.4 ; 1.5*2.5*0.4 2.5^2])
    sig = Matrix{Float64}(undef,i_dim,i_dim)
    means = Vector{Float64}(undef,i_dim)

    for j in 1:1:i_dim
        means[j] = 5+5*j
        sig[j,j] = (4*(1+j/n_dim[end]))^2 #
    end
    for j in 1:1:i_dim
        for k in 1:1:j
            sig[j,k] = 0.2*(1+i_dim/10)*sqrt(sig[j,j])*sqrt(sig[k,k])
            sig[k,j] = sig[j,k]
        end
    end

    dis = MvNormal(means,sig)
    iid_sample = bat_sample(dis, n_samples*n_chains).result;
    prior = NamedTupleDist(x = [Uniform(0,means[end]+5*sqrt(sig[end,end])) for i in 1:1:i_dim])
    mcmc_sample = bat_sample(PosteriorDensity(dis,prior), (n_samples, n_chains)).result
    results[i,2] =  round.(bat_compare(iid_sample, mcmc_sample).ks_p_values,digits=4)

    dis = BAT.MultimodalCauchy(µ=5. * i_dim, σ=0.2 * i_dim,n=i_dim)
    iid_sample = bat_sample(dis, n_samples*n_chains).result;
    prior = NamedTupleDist(x = [Uniform(-100,100) for i in 1:1:i_dim])
    mcmc_sample = bat_sample(PosteriorDensity(dis,prior), (n_samples, n_chains)).result
    results[i,3] =  round.(bat_compare(iid_sample, mcmc_sample).ks_p_values,digits=4)

    dis = BAT.FunnelDistribution(a=1., b=0.7, n=i_dim)
    iid_sample = bat_sample(dis, n_samples*n_chains).result;
    prior = NamedTupleDist(x = [Uniform(-100,100) for i in 1:1:i_dim])
    mcmc_sample = bat_sample(PosteriorDensity(dis,prior), (n_samples, n_chains)).result
    results[i,4] =  round.(bat_compare(iid_sample, mcmc_sample).ks_p_values,digits=4)
end

header = ["n_dims", "normal", "multimodal cauchy", "funnel"]

f = open("results/results_ND.txt","w")
pretty_table(f,results,header)
close(f)
fl = open("results/results_ND.tex","w")
pretty_table(fl,results,header,backend=:latex)
close(fl)
