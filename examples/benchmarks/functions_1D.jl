########1D functions for benchmarking###########
################################################
normal_dist =
	params -> begin
	   	return LogDVal(logpdf(Normal(0,1),params.x))
	end
prior_normal = NamedTupleDist(
	x = -7..7
)
posterior_normal = PosteriorDensity(normal_dist,prior_normal)
normal = (posterior=posterior_normal,mode=0,mean=0,var=1,chi2=[9999],ks=[9999.],ahmi=[9999.])

################################################
cauchy_dist =
	params -> begin
	   	return LogDVal(log(1/0.98)+logpdf(Cauchy(0,1),params.x))
	end
prior_cauchy = NamedTupleDist(
	x = -25..25
)
posterior_cauchy = PosteriorDensity(cauchy_dist,prior_cauchy)
cauchy = (posterior=posterior_cauchy,mode=0,mean=0,var=15.2459,chi2=[9999],ks=[9999.],ahmi=[9999.])

################################################
multi_cauchy_dist = let a=5., b=2.
	params -> begin
		return LogDVal(log(1/0.0212431856)+logpdf(BAT.MultimodalCauchy(µ=a,σ=b,n=1),[params.x,0]))
	end
end
prior_multi_cauchy = NamedTupleDist(
	x = -40..40
)
posterior_multi_cauchy = PosteriorDensity(multi_cauchy_dist,prior_multi_cauchy)
analytical_stats_multi_cauchy_variance = (2*(80 + 21/2*atan(35/2) + 21/2*atan(45/2) - 5*log(2029/1229)))/π
multi_cauchy = (posterior=posterior_multi_cauchy,mode=[-5,5],mean=0,var=analytical_stats_multi_cauchy_variance,chi2=[9999],ks=[9999.],ahmi=[9999.])

###############rastrigin x-cos(x)################
rastrigin_dist =
	params -> begin
		return LogDVal(BAT.logvalof(BAT.Rastrigin(),params.x))
	end

prior_rastrigin = NamedTupleDist(
	x = -3..3
)
posterior_rastrigin = PosteriorDensity(rastrigin_dist,prior_rastrigin)
rastrigin = (posterior=posterior_rastrigin,mode=0,mean=0,var=((219/85)+(5/(17*pi^2))),chi2=[9999],ks=[9999.],ahmi=[9999.])
################################################
sin2_dist =
	params -> begin
	   	return LogDVal(BAT.logvalof(BAT.SineSquared(),params.x))
	end

prior_sin2 = NamedTupleDist(
	x = 0..25
)
posterior_sin2 = PosteriorDensity(sin2_dist,prior_sin2)
analytical_mean_sin2 = -(5*(-195312509 + 2332509*cos(50) + 23250450*sin(50)))/47737968 #20.85992123409488
analytical_stats_sin2 = [23.6463,analytical_mean_sin2,12.0492]
sin2 = (posterior=posterior_sin2,mode=23.6463,mean=analytical_mean_sin2,var=12.0492,chi2=[9999],ks=[9999.],ahmi=[9999.])

################################################
exp1_dist = let a=1
    params -> begin
        return LogDVal(logpdf(Exponential(a),params.x))
    end
end
prior_exp1 = NamedTupleDist(
    x = 0..8
)
posterior_exp1 = PosteriorDensity(exp1_dist,prior_exp1)
exp1 = (posterior=posterior_exp1,mode=0,mean=1,var=1,chi2=[9999],ks=[9999.],ahmi=[9999.])

################################################
hoelder_dist =
    params -> begin
		return LogDVal(BAT.logvalof(HoelderTable(),params.x))
	end

prior_hoelder = NamedTupleDist(
    x = -10..10		#Fixed
)
posterior_hoelder = PosteriorDensity(hoelder_dist,prior_hoelder)
hoelder = (posterior=posterior_hoelder,mode=0,mean=0,var=23.0461,chi2=[9999],ks=[9999.],ahmi=[9999.])

################################################
analytical_integral_funnel = 1#0.2294350
funnel_dist = let a=1., b=0.5
    params -> begin
        return LogDVal(log(1/0.2294350)+logpdf(BAT.FunnelDistribution(a,b,2),[params.x,params.x]))
    end
end
prior_funnel = NamedTupleDist(
    x = -10..10
)
posterior_funnel = PosteriorDensity(funnel_dist,prior_funnel)
funnel = (posterior=posterior_funnel,mode=-0.298002,mean=0.0163721,var=0.314562,chi2=[9999],ks=[9999.],ahmi=[9999.])
#analytical refused by wolframalpha
################################################

testfunctions_1D = Dict(
	"normal" 		=> normal,
	"cauchy"		=> cauchy,
	"multi_cauchy"	=> multi_cauchy,
	"rastrigin"		=> rastrigin,
	"sin2"			=> sin2,
	"exp"			=> exp1,
	"hoelder"		=> hoelder,
	"funnel"		=> funnel
)

analytical_stats_name=["mode","mean","var"]	#could be taken into NamedTuple for easier addtions but would be needed to implmented into calcs anyway
sample_stats=[Vector{Float64}(undef,length(analytical_stats_name)) for i in 1:length(testfunctions_1D)]

run_stats_names = ["nsamples","nchains","Times"]
run_stats=[Vector{Float64}(undef,length(run_stats_names)) for i in 1:length(testfunctions_1D)]
