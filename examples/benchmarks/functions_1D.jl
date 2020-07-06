########1D functions for benchmarking###########
################################################
name_normal = "normal"
analytical_integral_normal = 1

normal =
	params -> begin
	   	return LogDVal(logpdf(Normal(0,1),params.x))
	end
prior_normal = NamedTupleDist(
	x = -7..7
)
posterior_normal = PosteriorDensity(normal,prior_normal)
analytical_function_normal = x -> pdf(Normal(0,1),x)
analytical_stats_normal = [0,0,1]#
################################################

################################################
name_cauchy = "cauchy"
analytical_integral_cauchy = 1#0.98

cauchy =
	params -> begin
	   	return LogDVal(logpdf(Cauchy(0,1),params.x))
	end
prior_cauchy = NamedTupleDist(
	x = -25..25
)
posterior_cauchy = PosteriorDensity(cauchy,prior_cauchy)
analytical_function_cauchy = x -> pdf(Cauchy(0,1),x)
analytical_stats_cauchy = [0,0,15.2459]#(50-2*atan(25))/pi] night ganz
################################################

################################################
name_multi_cauchy = "multi cauchy"
analytical_integral_multi_cauchy = 1#0.029053426970

multi_cauchy = let a=5., b=4.
	params -> begin
		mixture = [Cauchy(-1*a, b), Cauchy(a, b)]
		return LogDVal(logpdf(BAT.MultimodalCauchy(µ=a,σ=b,n=1),[params.x,0]))
	end
end
prior_multi_cauchy = NamedTupleDist(
	x = -40..40
)
posterior_multi_cauchy = PosteriorDensity(multi_cauchy,prior_multi_cauchy)
analytical_function_multi_cauchy = x-> pdf(BAT.MultimodalCauchy(µ=5.,σ=4.,n=1),[x,0])
analytical_stats_multi_cauchy_variance = (2*(80 + 21/2*atan(35/2) + 21/2*atan(45/2) - 5*log(2029/1229)))/π
analytical_stats_multi_cauchy = [[-5,5],0,analytical_stats_multi_cauchy_variance]
#Analytical Variance (2*(80 + 21/2*atan(35/2) + 21/2*atan(45/2) - 5*log(2029/1229)))/π
################################################

################rastrigin x-cos(x)##############
name_rastrigin = "rastrigin"
analytical_integral_rastrigin = 1

rastrigin =
	params -> begin
		return LogDVal(BAT.logvalof_unchecked(BAT.Rastrigin(),params.x))
	end

prior_rastrigin = NamedTupleDist(
	x = -3..3
)
posterior_rastrigin = PosteriorDensity(rastrigin,prior_rastrigin)
analytical_function_rastrigin = BAT.Rastrigin().dist
analytical_stats_rastrigin = [0,0,((219/85)+(5/(17*pi^2)))] #2.6062709363653935
################################################

################################################
name_sin2 = "sin2"
analytical_integral_sin2 = 1

sin2 =
	params -> begin
	   	return LogDVal(BAT.logvalof_unchecked(BAT.SineSquared(),params.x))
	end

prior_sin2 = NamedTupleDist(
	x = 0..25
)
posterior_sin2 = PosteriorDensity(sin2,prior_sin2)
analytical_function_sin2 = BAT.SineSquared().dist
analytical_mean_sin2 = -(5*(-195312509 + 2332509*cos(50) + 23250450*sin(50)))/47737968 #20.85992123409488
analytical_stats_sin2 = [23.6463,analytical_mean_sin2,12.0492]
################################################

################################################
name_exp1 = "exp1"
analytical_integral_exp1 = 1

exp1 = let a=1
    params -> begin
        return LogDVal(logpdf(Exponential(a),params.x))
    end
end
prior_exp1 = NamedTupleDist(
    x = 0..8
)
posterior_exp1 = PosteriorDensity(exp1,prior_exp1)
analytical_function_exp1 = x-> pdf(Exponential(1),x)
analytical_stats_exp1 = [0,1,1]
################################################

################################################
name_hoelder = "hoelder table"
analytical_integral_hoelder = 1

hoelder = let l=1
    params -> begin
		return LogDVal(BAT.logvalof_unchecked(HoelderTable(),params.x))
	end
end
prior_hoelder = NamedTupleDist(
    x = -10..10		#Fixed
)

posterior_hoelder = PosteriorDensity(hoelder,prior_hoelder)
analytical_function_hoelder = HoelderTable().dist#
analytical_stats_hoelder = [0,0,23.0461]
################################################

################################################
name_gaussian_shell = "gaussian shell"
analytical_integral_gaussian_shell = 1.988
gaussian_shell = let r=5, w=2, c=0
    params -> begin
        return LogDVal(log( ((1)/(sqrt(2*pi*w^2))) * exp(-((abs(params.x-c) - r)^2 / (2*w^2)) )  ))
    end
end
prior_gaussian_shell = NamedTupleDist(
    x = -25..25
)
posterior_gaussian_shell = PosteriorDensity(gaussian_shell,prior_gaussian_shell)
analytical_function_gaussian_shell = x->((1)/(sqrt(2*pi*2^2))) * exp(-((abs(x-0) - 5)^2 / (2*2^2)) )
analytical_stats_gaussian_shell = [[-5,5],0,29.1702] #analytical solution is really long
#29 + (5 sqrt(2/π))/e^(25/8) - 29/2 Q(1/2, 25/8) Q is regularized incomplete gamma function
################################################

################################################
name_funnel = "funnel"
analytical_integral_funnel = 1#0.2294350
funnel = let a=1., b=0.5
    params -> begin
        return LogDVal(logpdf(BAT.FunnelDistribution(a,b,2),[params.x,params.x]))
    end
end
prior_funnel = NamedTupleDist(
    x = -10..10
)
posterior_funnel = PosteriorDensity(funnel,prior_funnel)
analytical_function_funnel = x->pdf(BAT.FunnelDistribution(1.,0.5,2),[x,x])

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
            posterior_gaussian_shell,
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
            name_gaussian_shell,
            name_funnel
]

analytical_integral=[
			analytical_integral_normal,
            analytical_integral_cauchy,
            analytical_integral_multi_cauchy,
			analytical_integral_rastrigin,
			analytical_integral_sin2,
            analytical_integral_exp1,
            analytical_integral_hoelder,
            analytical_integral_gaussian_shell,
            analytical_integral_funnel
]

analytical_stats=[
            analytical_stats_normal,
            analytical_stats_cauchy,
            analytical_stats_multi_cauchy,
            analytical_stats_rastrigin,
            analytical_stats_sin2,
            analytical_stats_exp1,
            analytical_stats_hoelder,
            analytical_stats_gaussian_shell,
            analytical_stats_funnel
]

func=[
            analytical_function_normal,
            analytical_function_cauchy,
            analytical_function_multi_cauchy,
            analytical_function_rastrigin,
            analytical_function_sin2,
            analytical_function_exp1,
            analytical_function_hoelder,
            analytical_function_gaussian_shell,
            analytical_function_funnel
]

analytical_stats_name=["mode","mean","var"]
sample_stats=[Vector{Float64}(undef,length(analytical_stats_name)) for i in 1:length(name)]


run_stats_names = ["nsamples","nchains","Times"]
run_stats=[Vector{Float64}(undef,length(run_stats_names)) for i in 1:length(name)]
