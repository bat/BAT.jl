function plot_ks_values(results::Array{Any},n_dim;name="ks_test_values")
    n_diff_dims = length(n_dim)
    n_functions = length(create_testfunction_for_dim(1))

    x = Array{Array{Float64}}(undef,n_diff_dims*n_functions)
    y = Array{Array{Float64}}(undef,n_diff_dims*n_functions)
    x_dim_ticks_t1=Array{Float64}(undef,n_diff_dims*n_functions)
    x_dim_ticks_t2=Array{String}(undef,n_diff_dims*n_functions)

    xy_func_annot=Array{Array{Float64}}(undef,n_functions)
    func_annot = collect(keys(create_testfunction_for_dim(1)))

    for i_function in 1:n_functions
        for i_dim in 1:n_diff_dims
            index = i_dim+(i_function-1)*n_diff_dims
            y[index] = results[i_dim,1+i_function][1]
            x[index] = ones(length(y[index]))*(i_dim+(i_function-1)*(n_diff_dims+2))
            x_dim_ticks_t1[index]= (i_dim+(i_function-1)*(n_diff_dims+2))
            x_dim_ticks_t2[index]= string(n_dim[i_dim])
            if(i_dim == Int(floor((n_diff_dims/2)))+1 )
                xy_func_annot[i_function] = [(i_dim+(i_function-1)*(n_diff_dims+2)),-0.11]
            end
        end
    end

    scatter(x,y,label="",margin=5Plots.mm)
    for i in 1:length(xy_func_annot)
        annotate!(xy_func_annot[i]...,text(func_annot[i],10))
    end
    xticks!((x_dim_ticks_t1,x_dim_ticks_t2))
    savefig(name)
    savefig(string(name,"_ND.pdf"))
end

function plot_ahmi_values(results::Array{Any},n_dim;name="ahmi_values")
    n_diff_dims = length(n_dim)
    n_functions = length(create_testfunction_for_dim(1))

    x = Array{Float64}(undef,n_diff_dims*n_functions)
    y = Array{Float64}(undef,n_diff_dims*n_functions)
    yerr = Array{Float64}(undef,n_diff_dims*n_functions)


    x_ticks_t1=Array{Float64}(undef,n_diff_dims)
    x_ticks_t2=Array{String}(undef,n_diff_dims)
    y_ticks_t1=Array{Float64}(undef,n_functions*3)
    y_ticks_t2=Array{String}(undef,n_functions*3)
    xy_func_annot=Array{Array{Float64}}(undef,n_functions)
    func_annot = collect(keys(create_testfunction_for_dim(1)))

    int_plot_range = 0.2

    for i_function in 1:n_functions
     for i_dim in 1:n_diff_dims
         index = i_dim+(i_function-1)*n_diff_dims
         x[index] = i_dim
    	 y[index] = ((i_function-1)*(1))+(1-results[i_dim,1+i_function][2][1])
    	 yerr[index] = abs(results[i_dim,1+i_function][2][2])
         x_ticks_t1[i_dim] = i_dim
         x_ticks_t2[i_dim] = string(n_dim[i_dim])
     end

     y_ticks_t1[(i_function-1)*3+1]= ((i_function-1)*(1))-int_plot_range
     y_ticks_t1[(i_function-1)*3+2]= ((i_function-1)*(1))
     y_ticks_t1[(i_function-1)*3+3]= ((i_function-1)*(1))+int_plot_range
     y_ticks_t2[(i_function-1)*3+1]= string(1.0-int_plot_range)
     y_ticks_t2[(i_function-1)*3+2]= string(1.0)
     y_ticks_t2[(i_function-1)*3+3]= string(1.0+int_plot_range)
     xy_func_annot[i_function] = [1-0.55,((i_function-1)*(1))]
    end

    println(n_dim[1])
    println(n_dim[1])
    for i in 1:n_functions
    	i_start = n_diff_dims*(i-1)+1
    	i_end = n_diff_dims*i
        if i == 1
            plot(x[i_start:i_end],y[i_start:i_end],ribbon=yerr[i_start:i_end],label="",marker='x',margin=11Plots.mm)
        else
            plot!(x[i_start:i_end],y[i_start:i_end],ribbon=yerr[i_start:i_end],label="",marker='x',margin=11Plots.mm)
        end
    end

    for i in 1:length(xy_func_annot)
        annotate!(xy_func_annot[i]...,text(func_annot[i],9),annotation_clip=false)
    end
    #print(xy_func_annot)
    ylims!(-0.5,maximum(y)+0.3)
    xticks!((x_ticks_t1,x_ticks_t2))
    yticks!((y_ticks_t1,y_ticks_t2))
    xlabel!("Number of dimension")
    savefig(name)
    savefig(string(name,"_ND.pdf"))

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
        "cauchy" => cauchy,
        "funnel" => funnel
    )
end

function run_ND_benchmark(;n_dim = 2:2:20,algorithm = MetropolisHastings(),n_chains = 4,n_samples = 4*10^5)
    results = Matrix{Any}(undef,length(n_dim),length(create_testfunction_for_dim(1))+1)
    times = Matrix{Float64}(undef,length(n_dim),length(create_testfunction_for_dim(1))+1)

    convergence = BrooksGelmanConvergence(
        threshold = 1.6, #Change up
        corrected = false
    )

    init = MCMCInitStrategy(
        init_tries_per_chain = 64..1024,
        max_nsamples_init = 250,
        max_nsteps_init = 2500,
        max_time_init = Inf
    )

    burnin = MCMCBurninStrategy(
        max_nsamples_per_cycle = 5000,
        max_nsteps_per_cycle = 20000,
        max_time_per_cycle = Inf,
        max_ncycles = 300
    )

    for i in 1:length(n_dim)
        i_dim = n_dim[i]
        results[i,1] = i_dim

        testfunctions = create_testfunction_for_dim(i_dim,n_dim[end])

        for j in 1:length(testfunctions)
            print(i)
            print(j)
            dis = testfunctions[collect(keys(testfunctions))[j]]
            iid_sample = bat_sample(dis, n_samples*n_chains).result;
            tbf = time()
            mcmc_sample = bat_sample(
                dis, (n_samples, n_chains),algorithm,
                max_time = Inf,
                init = init,
                burnin = burnin,
                convergence = convergence,
                strict = true,
                filter = true
            ).result
            taf = time()
            integral = try
                bat_integrate(mcmc_sample,AHMIntegration()).result
            catch ex
                  NaN
            end
            if integral != NaN
                results[i,j+1] =  [round.(bat_compare(iid_sample, mcmc_sample).result.ks_p_values,digits=3),[round(integral.val,digits=3),round(integral.err,digits=3)]]
            else
                results[i,j+1] =  [round.(bat_compare(iid_sample, mcmc_sample).result.ks_p_values,digits=3),[NaN,NaN]]
            end
            times[i,j+1] = round(taf-tbf)
        end
    end

    header = ["n_dims",collect(keys(create_testfunction_for_dim(1)))...]
    f = open("results/results_ND.txt","w")
    pretty_table(f,results,header)
    close(f)
    fl = open("results/results_ND.tex","w")
    pretty_table(fl,results,header,backend=:latex)
    close(fl)

    header = ["n_dims",collect(keys(create_testfunction_for_dim(1)))...]
    t = open("results/times_ND.txt","w")
    pretty_table(t,times,header)
    close(t)
    tl = open("results/times_ND.tex","w")
    pretty_table(tl,times,header,backend=:latex)
    close(tl)

    plot_ks_values(results,n_dim)
    plot_ahmi_values(results,n_dim)
    return [results,times]
end
