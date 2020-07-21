using BAT, ValueShapes, IntervalSets, Distributions, Plots
using StatsBase, ArraysOfArrays, LinearAlgebra, LaTeXStrings, QuadGK, PrettyTables

function Statistics.cov(dist::BAT.FunnelDistribution)
    cov(nestedview(rand(BAT.bat_determ_rng(), sampler(dist), 10^5)))
end

function Statistics.cov(dist::BAT.MultimodalCauchy)
    cov(nestedview(rand(BAT.bat_determ_rng(), sampler(dist), 10^5)))
end

function create_testfunction_for_dim(i_dim::Integer,maxdim=10)
    sig = Matrix{Float64}([1.5^2 1.5*2.5*0.4 ; 1.5*2.5*0.4 2.5^2])
    sig = Matrix{Float64}(undef,i_dim,i_dim)
    means = Vector{Float64}(undef,i_dim)

    for j in 1:1:i_dim
        means[j] = 5+5*j
        sig[j,j] = (4*(1+j/maxdim))^2 #
    end
    for j in 1:1:i_dim
        for k in 1:1:j
            sig[j,k] = 0.2*(1+i_dim/10)*sqrt(sig[j,j])*sqrt(sig[k,k])
            sig[k,j] = sig[j,k]
        end
    end
    normal = MvNormal(means,sig)
    cauchy = BAT.MultimodalCauchy(µ=5. * i_dim, σ=0.2 * i_dim,n=i_dim)
    funnel = BAT.FunnelDistribution(a=1., b=0.7, n=i_dim)

    return Dict(
        "normal" => normal,
        #"cauchy" => cauchy,
        #"funnel" => funnel
    )
end



function run_ND_benchmark(;n_dim = 2:2:10,algorithm = MetropolisHastings(),n_chains = 10,n_samples = 10^4)
    results = Matrix{Any}(undef,length(n_dim),length(create_testfunction_for_dim(1))+1)

    for i in 1:length(n_dim)
        i_dim = n_dim[i]
        results[i,1] = i_dim

        testfunctions = create_testfunction_for_dim(i_dim,n_dim[end])

        for j in 1:length(testfunctions)
            dis = testfunctions[collect(keys(testfunctions))[j]]
            iid_sample = bat_sample(dis, n_samples*n_chains).result;
            mcmc_sample = bat_sample(dis, (n_samples, n_chains),algorithm).result
            integral = bat_integrate(mcmc_sample,AHMIntegration()).result.val
            results[i,j+1] =  [round.(bat_compare(iid_sample, mcmc_sample).result.ks_p_values,digits=3),round(integral,digits=3)]
        end
    end

    header = ["n_dims",collect(keys(create_testfunction_for_dim(1)))...]
    f = open("results/results_ND.txt","w")
    pretty_table(f,results,header)
    close(f)
    fl = open("results/results_ND.tex","w")
    pretty_table(fl,results,header,backend=:latex)
    close(fl)
end
