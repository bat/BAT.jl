#1D functions
function multimodal_1D(sample_sats::Vector{Vector{Float64}},analytical_stats::Vector{Vector{Any}})
    new_analytical_stats = []
	print(analytical_stats)
	print(sample_sats)
    for i in 1:length(sample_stats)
        push!(new_analytical_stats,one_multimodal_1D(sample_stats[i],analytical_stats[i]))
    end
    return new_analytical_stats
end

function one_multimodal_1D(sample_stats::Any,analytical_stats::Any)
    new_analytical_stats = Array{Float64}(undef,length(analytical_stats))
    diff = abs.(sample_stats[1])
    if length(analytical_stats[1]) > 1
        new_analytical_stats[1] = analytical_stats[1][argmin(abs.(analytical_stats[1] .- sample_stats[1]))]
        for i in 2:length(analytical_stats)
            new_analytical_stats[i] = analytical_stats[i]
        end
        return new_analytical_stats
    else
        return analytical_stats
    end
end

function plot_diff_1D(
	bincenters::Vector{Float64},
	differences::Vector{Float64},
	err1::Vector{Float64},
	err2::Vector{Float64},
	err3::Vector{Float64},
	name::String)

    scatter(
        bincenters,
        differences,
        marker=(:x,2),
        label="Difference",
        ylim=(-maximum(vcat(err3...,abs.(differences)...)),maximum(vcat(err3...,abs.(differences)...))),
        ylabel=L"$f_{MCMC}-f_{ana}$"
    )
    nfill=0.3
    plot!(
        bincenters,
        err1,
        fillrange = 0,
        color="green",
        label="",
        fillalpha=nfill
    )
    plot!(
        bincenters,
        -err1,
        fillrange = 0,
        color="green",
        label="",
        fillalpha=nfill
    )
    plot!(
        bincenters,
        err2,
        fillrange = err1,
        color="yellow",
        label="",
        fillalpha=nfill
    )
    plot!(
        bincenters,
        -err2,
        fillrange = -err1,
        color="yellow",
        label="",
        fillalpha=nfill
    )
    plot!(
        bincenters,
        err3,
        fillrange = err2,
        color="red",
        label="",
        fillalpha=nfill
    )
    plot!(
        bincenters,
        -err3,
        fillrange = -err2,
        color="red",
        label="",
        fillalpha=nfill
    )
    savefig(string("plots1D/",name,"_diff.pdf"))
end

function plot1D(
	samples::DensitySampleVector,
	testfunctions::Dict,
	name::String,
	sample_stats::Vector{Float64})

	func = k->pdf(testfunctions[name].posterior,[k])[1]

	lo = minimum([BAT.flatview(samples.v)...])
	hi = maximum([BAT.flatview(samples.v)...])
	if(!isa(testfunctions_1D[name].posterior, BAT.FunnelDistribution))
		lo, hi = quantile.(testfunctions_1D[name].posterior, [0.001, 0.999]) #BAT Distributions do not support quantiles
	end
	if(isa(testfunctions_1D[name].posterior, Distributions.Cauchy))
		lo, hi = quantile.(testfunctions_1D[name].posterior, [0.01, 0.99]) #Cauchy needs smaller interval for useful plots however just for the plot
	end
	binning = range(lo, hi; length=200)

    hunnorm = fit(Histogram, [BAT.flatview(samples.v)...],FrequencyWeights(BAT.flatview(samples.weight)),binning)
	h = fit(Histogram, [BAT.flatview(samples.v)...],FrequencyWeights(BAT.flatview(samples.weight)),binning)
	idx = BAT.asindex(samples, 1)
	uvbd = EmpiricalDistributions.UvBinnedDist(h)
	marg = BAT.MarginalDist((idx,),uvbd,varshape(samples))
	h = StatsBase.normalize(h)

	plot(marg,1,seriestype = :smallest_intervals,normalize=true)

    edge_widths = h.edges[1][2:end]-h.edges[1][1:end-1]
    edge_mids = h.edges[1][1:end-1].+ (0.5*edge_widths)

	plot!(edge_mids,broadcast(func,edge_mids),label="Analytical")
	savefig(string("plots1D/",name,".pdf"))

	if(isa(testfunctions_1D[name].posterior, Distributions.Cauchy))
		lo, hi = quantile.(testfunctions_1D[name].posterior, [0.001, 0.999])
		binning = range(lo, hi; length=200)
	end

    nun = convert(Int64,floor(sum(hunnorm.weights)/10))
    unweighted_samples = bat_sample(samples, nun).result
    hunnorm = fit(Histogram, [BAT.flatview(unweighted_samples.v)...],binning)

    edges = hunnorm.edges[1]
    nbins = length(edges)-1
    bincenters = zeros(nbins)
    differences = zeros(nbins)
    err1 = zeros(nbins)
    err2 = zeros(nbins)
    err3 = zeros(nbins)
    pulls = zeros(nbins)
    norm_hist = sum(hunnorm.weights)

    ndf = nbins
    chi2 = 0

    narr = zeros(nbins)
    earr = zeros(nbins)

    for ibin in 1:nbins
        bincenters[ibin] = edges[ibin]+0.5*(edges[ibin+1]-edges[ibin])
        n = hunnorm.weights[ibin]
        e = norm_hist * quadgk(func,edges[ibin],edges[ibin+1])[1]
        narr[ibin] = n
        earr[ibin] = e
        differences[ibin] = n - e
        err1[ibin] = sqrt(e)
        err2[ibin] = 2 * err1[ibin]
        err3[ibin] = 3 * err1[ibin]
        pulls[ibin] = (n-e)/sqrt(e)
        if (e >= 10)
            chi2 += (n - e) * (n - e) / e
        else
            ndf = ndf-1
        end
    end

    plot_diff_1D(bincenters,differences,err1,err2,err3,name)

    hpull = fit(Histogram,pulls[pulls .!= 999],nbins=25)
    d = Normal(0,1)
    lo, hi = quantile.(d, [0.00001, 0.99999])
    x = range(lo, hi; length = 100)
    binlength = hpull.edges[1].step.hi
    plot(hpull,st=:step,normalize=false,label="",xlabel=L"$\Delta/\sigma$")
    plot!(x, pdf.(d, x)*sum(hpull.weights)*binlength,label="")
    savefig(string("plots1D/",name,"_pull.pdf"))

    if(testfunctions[name].chi2[1] == 9999)
		testfunctions[name].chi2[1]=ndf
	end
    length(sample_stats) != 4 ? push!(sample_stats,chi2)     : sample_stats[4] = chi2

	iid_sample = bat_sample(testfunctions[name].posterior,length([BAT.flatview(samples.v)...])).result
	if(testfunctions[name].ks[1] > 999)
		testfunctions[name].ks[1]=bat_compare(samples,iid_sample).result.ks_p_values[1]
	end

	if(testfunctions[name].ahmi[1] > 999)
		testfunctions[name].ahmi[1]=bat_integrate(samples,AHMIntegration()).result.val
	end

	return h
end

function run1D(
	key::String,
	testfunctions::Dict,
    sample_stats::Vector{Float64},
    run_stats::Vector{Float64},
	algorithm::BAT.AbstractSamplingAlgorithm,
	n_samples::Integer,
	n_chains::Integer,
    n_runs=1
	)

    sample_stats_all = []
    samples, chains = bat_sample(testfunctions[key].posterior, n_samples * n_chains, MCMCSampling(sampler = algorithm, nchains = n_chains))
    for i in 1:n_runs
        time_before = time()
        samples, chains = bat_sample(testfunctions[key].posterior, n_samples * n_chains, MCMCSampling(sampler = algorithm, nchains = n_chains))
        time_after = time()

    	h = plot1D(samples,testfunctions,key,sample_stats)# posterior, key, analytical_stats,sample_stats)

        sample_stats[1] = mode(samples)[1]
        sample_stats[2] = mean(samples)[1]
        sample_stats[3] = var(samples)[1]
        run_stats[1] = n_samples
        run_stats[2] = n_chains
        run_stats[3] = time_after-time_before
        push!(sample_stats_all,sample_stats)
    end
	return sample_stats_all
end

function make_1D_results(
	testfunctions::Dict,
	sample_stats::Vector{Vector{Float64}})
	#analytical_stats::Vector{Vector{Any}})

	name = Vector{String}()
	analytical_stats = Vector{Vector{Any}}()
	ks_p_val = Vector{Float64}()
	ahmi_val = Vector{Float64}()
	for (k,v) in testfunctions
		push!(name,k)
		push!(analytical_stats,[v.mode,v.mean,v.var,v.chi2[1]])
		push!(ks_p_val,round(v.ks[1],digits=3))
		push!(ahmi_val,round(v.ahmi[1],digits=3))
	end
    statistics_names = ["mode","mean","var","chi2"]
    comparison = ["target","test","diff (abs)","diff (rel)"]
    analytical_stats = multimodal_1D(sample_stats,analytical_stats)
    sample_stats = round.(permutedims(reshape(vcat(sample_stats...),length(sample_stats[1]),length(sample_stats))),digits=3)
    analytical_stats = round.(permutedims(reshape(vcat(analytical_stats...),length(analytical_stats[1]),length(analytical_stats))),digits=3)
    diff_stats_abs = round.(sample_stats-analytical_stats,digits=3)
    diff_stats_rel = round.(diff_stats_abs ./ analytical_stats * 100,digits=3)
    comparison_vals = [analytical_stats,sample_stats,diff_stats_abs,diff_stats_rel]

    #final_header = Vector{Any}(undef,length(statistics_names)*length(comparison)+1)
	#final_table = Array{Any}(undef,length(name),length(statistics_names)*length(comparison)+1)
	final_header = Vector{Any}(undef,length(name)+1)
	final_table = Array{Any}(undef,length(statistics_names)*length(comparison)+1+1,length(name)+1)
    final_header[1] = "name"
    for j in 1:length(statistics_names)
        for k in 1:length(comparison)
            final_table[(j-1)*length(statistics_names)+k,1] = string(statistics_names[j]," ",comparison[k])
        end
    end
    final_header[2:end]=name
    for i in 1:length(name)
        for j in 1:length(statistics_names)
            final_table[(j-1)*length(statistics_names)+1,i+1] = analytical_stats[i,j]
            final_table[(j-1)*length(statistics_names)+2,i+1] = sample_stats[i,j]
            final_table[(j-1)*length(statistics_names)+3,i+1] = diff_stats_abs[i,j]
            final_table[(j-1)*length(statistics_names)+4,i+1] = diff_stats_rel[i,j]
        end
    end

	final_table[length(statistics_names)*length(comparison)+1,:] = ["KS test p value",ks_p_val...]
	final_table[length(statistics_names)*length(comparison)+2,:] = ["AHMI Integral",ahmi_val...]
	f = open("results/results_1D.txt","w")
    pretty_table(f,final_table,final_header)
    close(f)
	fl = open("results/results_1D.tex","w")
    pretty_table(fl,final_table,final_header,backend=:latex)
    close(fl)
end

function save_stats_1D(name::Vector{String},
	run_stats::Vector{Vector{Float64}},
	run_stats_names::Vector{String})

    run_stats_table = reshape(vcat(run_stats...),length(run_stats[1]),length(run_stats))
    table_stats = Any[]
    append!(table_stats,[name])
    for i in 1:length(run_stats_table[:,1])
        append!(table_stats,[run_stats_table[i,:]])
    end
    table_stats = reshape(vcat(table_stats...),length(table_stats[1]),length(table_stats))
	table_stats[1:end,4] = round.(table_stats[1:end,4],digits=0)
	table_stats[1:end,2:end] = Int.(table_stats[1:end,2:end])
	f = open("results/run_stats_1D.txt","w")
    pretty_table(f,table_stats,["name",run_stats_names...])
    close(f)
	fl = open("results/run_stats_1D.tex","w")
    pretty_table(fl,table_stats,["name",run_stats_names...],backend=:latex)
    close(fl)
end

#2D functions

function plot2D(
	samples::DensitySampleVector,
	testfunctions::Dict,
	name::String,
	sample_stats::Vector{Any})

    nbin = 400

	plot_bins = nothing
	if name == "funnel"
		plot_bins = (-5:0.1:5,-10:0.1:10)
	elseif name == "multi cauchy"
		plot_bins = (-200:0.4:200,-200:0.4:200)
	else
		plot_bins = (5:0.1:25,-5:0.1:25)
	end

    h = fit(Histogram, (BAT.flatview(samples.v)[1,:],BAT.flatview(samples.v)[2,:]) ,FrequencyWeights(samples.weight),plot_bins)
    hunnorm = fit(Histogram, (BAT.flatview(samples.v)[1,:],BAT.flatview(samples.v)[2,:]) ,FrequencyWeights(samples.weight),plot_bins)


    h = StatsBase.normalize(h)

    nun = convert(Int64,floor(sum(hunnorm.weights)/10))
    unweighted_samples = bat_sample(samples, nun).result
    hunnorm = fit(Histogram, (BAT.flatview(unweighted_samples.v)[1,:],BAT.flatview(unweighted_samples.v)[2,:]),plot_bins)
    hana = fit(Histogram,([],[]),hunnorm.edges)
    hdiff = fit(Histogram,([],[]),hunnorm.edges)

    nbinx = length(hunnorm.edges[1])-1
    nbiny = length(hunnorm.edges[2])-1
    differences = zeros(nbinx,nbiny)
    pulls = zeros(nbinx,nbiny)


    ndf = nbinx*nbiny
    chi2 = 0
	v = Vector{Array{Float64,1}}()
	logval = Vector{Float64}()
	w = Vector{Float64}()


    for i in 1:length(hunnorm.edges[1])-1
        for j in 1:length(hunnorm.edges[2])-1
            binmidx = (hunnorm.edges[1][i+1]-hunnorm.edges[1][i])/2 + hunnorm.edges[1][i]
            binmidy = (hunnorm.edges[2][j+1]-hunnorm.edges[2][j])/2 + hunnorm.edges[2][j]
            binarea = (hunnorm.edges[1][i+1]-hunnorm.edges[1][i]) * (hunnorm.edges[2][j+1]-hunnorm.edges[2][j])

			#anaval = exp(posterior.likelihood.f((x=binmidx,y=binmidy)).value)*sum(hunnorm.weights)*binarea
			anaval = pdf(testfunctions[name].posterior,[binmidx,binmidy])*sum(hunnorm.weights)*binarea

            hana.weights[i,j] = convert(Int64,floor(anaval+0.5))

            differences[i,j] = (anaval-hunnorm.weights[i,j])
            hana.weights[i,j] < 10 ?  pulls[i,j] = 999 : pulls[i,j] = differences[i,j]/sqrt(anaval)

            hdiff.weights[i,j] = convert(Int64,floor(differences[i,j]+0.5))

            if (hunnorm.weights[i,j] >= 10)
            chi2 += (hunnorm.weights[i,j] - hana.weights[i,j]) * (hunnorm.weights[i,j] - hana.weights[i,j]) / hana.weights[i,j]
            else
                ndf = ndf-1
            end

			push!(v,[binmidx,binmidy])
			push!(logval,anaval)
			push!(w,anaval)
        end
    end


	if(testfunctions[name].chi2[1] == 9999)
		testfunctions[name].chi2[1]=ndf
	end

	iid_sample = bat_sample(testfunctions[name].posterior,length([BAT.flatview(samples.v)...])).result
	if(testfunctions[name].ks[1] > 999)
		ksres = bat_compare(samples,iid_sample).result.ks_p_values
		testfunctions[name].ks[1]=ksres[1]
		testfunctions[name].ks[2]=ksres[2]
	end

	if(testfunctions[name].ahmi[1] > 999)
		ahmires = bat_integrate(samples,AHMIntegration()).result
		testfunctions[name].ahmi[1]=ahmires.val
		testfunctions[name].ahmi[2]=ahmires.err
	end

	#plot(hunnorm,(1,2),seriestype=:smallest_intervals)

	if name == "multi cauchy"
		plot_bins = (-60:0.5:60,-60:0.5:60)
		plot(samples,bins=plot_bins,globalmode=false)
		savefig(string("plots2D/default_",name,".pdf"))
	else
		plot(samples,bins=plot_bins,globalmode=true)
		savefig(string("plots2D/default_",name,".pdf"))
	end

	plot(unweighted_samples,(1,2),seriestype=:smallest_intervals,bins=plot_bins)
    savefig(string("plots2D/",name,".pdf"))

    plot(hana)#,(1,2),seriestype=:smallest_intervals)
    savefig(string("plots2D/",name,"_analytic.pdf"))

	#plot(hdiff,(1,2))
	plot(hdiff)
    savefig(string("plots2D/",name,"_diff.pdf"))

    pulls = vcat(pulls...)

    hpull = fit(Histogram,pulls[pulls .!= 999],nbins=30)
    d = Normal(0,1)
    lo, hi = quantile.(d, [0.00001, 0.99999])
    x = range(lo, hi; length = 100)
    binlength = hpull.edges[1].step.hi
    plot(hpull,normalize=false,label="",xlabel=L"$\Delta/\sigma$",ylabel="N",xlims=(-5,5))
    plot!(x, pdf.(d, x)*sum(hpull.weights)*binlength,label="",linewidth=2)
    savefig(string("plots2D/",name,"_pull.pdf"))

    return h
end

function run2D(
	key::String,
	testfunctions::Dict,
    sample_stats::Vector{Any},
    run_stats::Vector{Any},
	algorithm::MCMCAlgorithm,
	n_samples::Integer,
	n_chains::Integer,
	n_runs=1)

    sample_stats_all = []

    samples, stats = bat_sample(testfunctions[key].posterior, n_samples * n_chains, MCMCSampling(sampler = algorithm, nchains = n_chains))
    for i in 1:n_runs
        time_before = time()
        samples, stats = bat_sample(testfunctions[key].posterior, n_samples * n_chains, MCMCSampling(sampler = algorithm, nchains = n_chains))
        time_after = time()

		h = plot2D(samples, testfunctions, key, sample_stats)

        sample_stats[1] = mode(samples).data
        sample_stats[2] = mean(samples).data
        sample_stats[3] = var(samples).data

        run_stats[1] = n_samples
        run_stats[2] = n_chains
        run_stats[3] = time_after-time_before
        push!(sample_stats_all,sample_stats)
    end

    return sample_stats_all
end

function make_2D_results(testfunctions::Dict,sample_stats2D::Vector{Vector{Any}})

	name2D = Vector{String}()
	analytical_stats2D = Vector{Vector{Any}}()
	ks_p_val = Vector{Any}()
	ahmi_val = Vector{Any}()

	for (k,v) in testfunctions
		push!(name2D,k)
		push!(analytical_stats2D,[v.mode,v.mean,v.var])
		push!(ks_p_val,round.(v.ks,digits=3))
		push!(ahmi_val,round.(v.ahmi,digits=3))
	end

    run_stats_names2D = ["nsamples","nchains","Times"]
    stats_names2D = ["mode","mean","var"]
    comparison = ["target","test","diff (abs)","diff (rel)"]
    header  = Vector{Any}(undef,length(stats_names2D)*length(comparison)+3)
    table   = Array{Any}(undef,length(name2D),length(stats_names2D)*length(comparison)+3)
    header[1] = "name"
    table[1:end,1] = name2D

    for i in 1:length(name2D)
        for j in 1:length(stats_names2D)
            header[1+(j-1)*length(comparison)+1]    = string(stats_names2D[j]," ",comparison[1])
            header[1+(j-1)*length(comparison)+2]  = string(stats_names2D[j]," ",comparison[2])
            header[1+(j-1)*length(comparison)+3]  = string(stats_names2D[j]," ",comparison[3])
            header[1+(j-1)*length(comparison)+4]  = string(stats_names2D[j]," ",comparison[4])
            sample_stat = sample_stats2D[i][j]
            analytical_stat = analytical_stats2D[i][j]
            if ((length(analytical_stat[1]) != length(sample_stat[1]) ) && stats_names2D[j] == "mode")
                diffs = Array{Float64}(undef,length(analytical_stat))
                for k in 1:length(analytical_stat)
                    diffs[k] = sum(abs.(analytical_stat[k] .- sample_stat))
                end
                analytical_stat  = analytical_stat[argmin(diffs)]
            end
            diff_abs = analytical_stat.-sample_stat
            diff_rel = diff_abs./analytical_stat
            table[i,1+(j-1)*length(comparison)+1]  =  round.(analytical_stat,digits=3)
            table[i,1+(j-1)*length(comparison)+2]  =  round.(sample_stat,digits=3)
            table[i,1+(j-1)*length(comparison)+3]  =  round.(diff_abs,digits=3)
            table[i,1+(j-1)*length(comparison)+4]  =  round.(diff_rel,digits=3)
        end
		table[i,length(stats_names2D)*length(comparison)+2] = ks_p_val[i]
		table[i,length(stats_names2D)*length(comparison)+3] = ahmi_val[i]
    end

	header[length(stats_names2D)*length(comparison)+2] = "KS test p-value"
	header[length(stats_names2D)*length(comparison)+3] = "AHMI integral"
	full_table = Array{Any}(undef,length(name2D)+1,length(stats_names2D)*length(comparison)+3)
	full_table[1,:] = header
	full_table[2:end,:] = table
	full_table = permutedims(full_table)
	table = full_table[2:end,:]
	header = full_table[1,:]
	f = open("results/results_2D.txt","w")
    pretty_table(f,table,header)
    close(f)
	fl = open("results/results_2D.tex","w")
    pretty_table(fl,table,header,backend=:latex)
    close(fl)
end

function save_stats_2D(name::Vector{String},run_stats::Vector{Vector{Any}},run_stats_names::Vector{String})
    run_stats_table = reshape(vcat(run_stats...),length(run_stats[1]),length(run_stats))
    table_stats = Any[]
    append!(table_stats,[name])
    for i in 1:length(run_stats_table[:,1])
        append!(table_stats,[run_stats_table[i,:]])
    end
    table_stats = reshape(vcat(table_stats...),length(table_stats[1]),length(table_stats))
	table_stats[1:end,4] = round.(table_stats[1:end,4],digits=0)
	table_stats[1:end,2:end] = Int.(table_stats[1:end,2:end])
	f = open("results/run_stats_2D.txt","w")
    pretty_table(f,table_stats,["name",run_stats_names...])
    close(f)
	fl = open("results/run_stats_2D.tex","w")
    pretty_table(fl,table_stats,["name",run_stats_names...],backend=:latex)
    close(fl)
end
