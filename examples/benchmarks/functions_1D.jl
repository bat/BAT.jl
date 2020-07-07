########1D functions for benchmarking###########
################################################
name_normal = "normal"

normal =
	params -> begin
	   	return LogDVal(logpdf(Normal(0,1),params.x))
	end
prior_normal = NamedTupleDist(
	x = -7..7
)
posterior_normal = PosteriorDensity(normal,prior_normal)
analytical_stats_normal = [0,0,1]#
################################################

################################################
name_cauchy = "cauchy"

cauchy =
	params -> begin
	   	return LogDVal(log(1/0.98)+logpdf(Cauchy(0,1),params.x))
	end
prior_cauchy = NamedTupleDist(
	x = -25..25
)
posterior_cauchy = PosteriorDensity(cauchy,prior_cauchy)
analytical_stats_cauchy = [0,0,15.2459]#(50-2*atan(25))/pi] night ganz
################################################

################################################
name_multi_cauchy = "multi cauchy"

multi_cauchy = let a=5., b=2.
	params -> begin
		return LogDVal(log(1/0.0212431856)+logpdf(BAT.MultimodalCauchy(µ=a,σ=b,n=1),[params.x,0])) #Normalize Cauchy for -40,40 prior to 1
	end
end
prior_multi_cauchy = NamedTupleDist(
	x = -40..40
)
posterior_multi_cauchy = PosteriorDensity(multi_cauchy,prior_multi_cauchy)
analytical_stats_multi_cauchy_variance = (2*(80 + 21/2*atan(35/2) + 21/2*atan(45/2) - 5*log(2029/1229)))/π
analytical_stats_multi_cauchy = [[-5,5],0,analytical_stats_multi_cauchy_variance]
#Analytical Variance (2*(80 + 21/2*atan(35/2) + 21/2*atan(45/2) - 5*log(2029/1229)))/π
################################################

################rastrigin x-cos(x)##############
name_rastrigin = "rastrigin"

rastrigin =
	params -> begin
		return LogDVal(BAT.logvalof(BAT.Rastrigin(),params.x))
	end

prior_rastrigin = NamedTupleDist(
	x = -3..3
)
posterior_rastrigin = PosteriorDensity(rastrigin,prior_rastrigin)
analytical_stats_rastrigin = [0,0,((219/85)+(5/(17*pi^2)))] #2.6062709363653935
################################################

################################################
name_sin2 = "sin2"

sin2 =
	params -> begin
	   	return LogDVal(BAT.logvalof(BAT.SineSquared(),params.x))
	end

prior_sin2 = NamedTupleDist(
	x = 0..25
)
posterior_sin2 = PosteriorDensity(sin2,prior_sin2)
analytical_mean_sin2 = -(5*(-195312509 + 2332509*cos(50) + 23250450*sin(50)))/47737968 #20.85992123409488
analytical_stats_sin2 = [23.6463,analytical_mean_sin2,12.0492]
################################################

################################################
name_exp1 = "exp1"

exp1 = let a=1
    params -> begin
        return LogDVal(logpdf(Exponential(a),params.x))
    end
end
prior_exp1 = NamedTupleDist(
    x = 0..8
)
posterior_exp1 = PosteriorDensity(exp1,prior_exp1)
analytical_stats_exp1 = [0,1,1]
################################################

################################################
name_hoelder = "hoelder table"

hoelder = let l=1
    params -> begin
		return LogDVal(BAT.logvalof(HoelderTable(),params.x))
	end
end
prior_hoelder = NamedTupleDist(
    x = -10..10		#Fixed
)
posterior_hoelder = PosteriorDensity(hoelder,prior_hoelder)
analytical_stats_hoelder = [0,0,23.0461]
################################################

################################################
name_funnel = "funnel"

analytical_integral_funnel = 1#0.2294350
funnel = let a=1., b=0.5
    params -> begin
        return LogDVal(log(1/0.2294350)+logpdf(BAT.FunnelDistribution(a,b,2),[params.x,params.x]))
    end
end
prior_funnel = NamedTupleDist(
    x = -10..10
)
posterior_funnel = PosteriorDensity(funnel,prior_funnel)
analytical_stats_funnel = [-0.298002,0.0163721,0.314562] #analytical refused by wolframalpha
################################################

posterior=[
            posterior_normal,
            posterior_cauchy,
            posterior_multi_cauchy,
			posterior_rastrigin,
            posterior_sin2,
            posterior_exp1,
            posterior_hoelder,
            posterior_funnel
		]

name=[
            name_normal,
            name_cauchy,
            name_multi_cauchy,
			name_rastrigin,
			name_sin2,
            name_exp1,
            name_hoelder,
            name_funnel
]

analytical_stats=[
            analytical_stats_normal,
            analytical_stats_cauchy,
            analytical_stats_multi_cauchy,
            analytical_stats_rastrigin,
            analytical_stats_sin2,
            analytical_stats_exp1,
            analytical_stats_hoelder,
            analytical_stats_funnel
]

analytical_stats_name=["mode","mean","var"]
sample_stats=[Vector{Float64}(undef,length(analytical_stats_name)) for i in 1:length(name)]

run_stats_names = ["nsamples","nchains","Times"]
run_stats=[Vector{Float64}(undef,length(run_stats_names)) for i in 1:length(name)]
